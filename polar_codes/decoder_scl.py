"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def _llr_to_bit(self, llr):
        return 0 if llr >= 0 else 1

    def _pm_update(self, pm, llr, u):
        bit = self._llr_to_bit(llr)
        if u == bit:
            return pm
        return pm + abs(llr)

    def _update_llrs(self, L, B, leaf):
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(leaf, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(leaf, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, leaf, u_bit):
        n = self.n
        N = self.N
        B[leaf, n] = u_bit
        if leaf < N // 2:
            return
        for s in range(n, n - _active_bit_level(leaf, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(leaf, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        L_size = self.list_size

        llr = np.asarray(llr_ch, dtype=np.float64)
        paths = [
            {
                "pm": 0.0,
                "u_hat": np.zeros(N, dtype=int),
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
            }
        ]
        paths[0]["L"][:, 0] = llr

        for phi in range(N):
            leaf = self.br[phi]
            new_paths = []

            for path in paths:
                self._update_llrs(path["L"], path["B"], leaf)
                cur_llr = path["L"][leaf, n]

                if self.frozen_bits[phi]:
                    child = {
                        "pm": self._pm_update(path["pm"], cur_llr, 0),
                        "u_hat": path["u_hat"].copy(),
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                    }
                    child["u_hat"][phi] = 0
                    self._update_bits(child["B"], leaf, 0)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = {
                            "pm": self._pm_update(path["pm"], cur_llr, u),
                            "u_hat": path["u_hat"].copy(),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        child["u_hat"][phi] = u
                        self._update_bits(child["B"], leaf, u)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:L_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p["pm"])

        return best["u_hat"], best["pm"]
