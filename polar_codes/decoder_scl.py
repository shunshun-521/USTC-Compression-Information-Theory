"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.frozen_set = set(int(self.br[i]) for i in np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _llr_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _compute_llr(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )
        return L[l, self.n]

    def _update_bits(self, B, l, bit):
        B[l, self.n] = bit
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        decode_order = [int(self.br[i]) for i in range(self.N)]

        paths = []
        L0 = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B0 = np.full((self.N, self.n + 1), np.nan)
        L0[:, 0] = llr_ch
        paths.append({"pm": 0.0, "L": L0, "B": B0, "u_scd": np.zeros(self.N, dtype=int)})

        for l in decode_order:
            new_paths = []
            for path in paths:
                llr_val = self._compute_llr(path["L"], path["B"], l)
                if l in self.frozen_set:
                    child = {
                        "pm": path["pm"] + self._llr_penalty(llr_val, 0),
                        "L": path["L"],
                        "B": path["B"].copy(),
                        "u_scd": path["u_scd"].copy(),
                    }
                    child["u_scd"][l] = 0
                    self._update_bits(child["B"], l, 0)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = {
                            "pm": path["pm"] + self._llr_penalty(llr_val, bit),
                            "L": path["L"],
                            "B": path["B"].copy(),
                            "u_scd": path["u_scd"].copy(),
                        }
                        child["u_scd"][l] = bit
                        self._update_bits(child["B"], l, bit)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_hat = p["u_scd"][self.br]
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p["pm"])

        u_hat = best["u_scd"][self.br]
        return u_hat, best["pm"]
