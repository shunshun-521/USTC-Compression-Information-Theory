"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
    _frozen_indices_from_mask,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length)[-crc_length:], bits[-crc_length:])


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 B 矩阵，L 矩阵各路径独立更新）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(_frozen_indices_from_mask(frozen_bits))
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.info_indices = np.where(np.asarray(frozen_bits, dtype=int) == 0)[0]

    def _init_path(self, llr_perm):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_perm
        return {"L": L, "B": B, "pm": 0.0, "u_hat": np.zeros(self.N, dtype=int)}

    def _pm_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return abs(llr) if u_bit != hard else 0.0

    def _update_llrs(self, path, l):
        L = path["L"]
        B = path["B"]
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch, s], B[j - branch, s + 1]
                    )

    def _update_bits(self, path, l):
        B = path["B"]
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch:
                    B[j - branch, s - 1] = B[j, s] ^ B[j - branch, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = llr_ch[self.br]
        paths = [self._init_path(llr_perm)]

        for phase in range(self.N):
            l = _bit_reversed_index(phase, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_leaf = path["L"][l, self.n]

                if l in self.frozen_set:
                    child = path
                    child = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": path["pm"] + self._pm_penalty(llr_leaf, 0),
                        "u_hat": path["u_hat"].copy(),
                    }
                    child["B"][l, self.n] = 0
                    child["u_hat"][l] = 0
                    self._update_bits(child, l)
                    candidates.append(child)
                else:
                    bit_order = (0, 1) if llr_leaf >= 0 else (1, 0)
                    if self.list_size == 1:
                        bit_order = (bit_order[0],)
                    for u_bit in bit_order:
                        child = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": path["pm"] + self._pm_penalty(llr_leaf, u_bit),
                            "u_hat": path["u_hat"].copy(),
                        }
                        child["B"][l, self.n] = u_bit
                        child["u_hat"][l] = u_bit
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: (p["pm"], p["u_hat"].sum()))
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            for path in paths:
                payload = path["u_hat"][self.info_indices]
                if crc_check(payload, self.crc_length):
                    return path["u_hat"], path["pm"]
        best = paths[0]
        return best["u_hat"], best["pm"]
