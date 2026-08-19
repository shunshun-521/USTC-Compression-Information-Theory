"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from channel import reorder_llr_for_decode
from decoder_sc import (
    _frozen_set,
    active_bit_level,
    active_llr_level,
    bit_reversed,
    lower_llr,
    upper_llr,
)


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_polynomial(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length <= 0:
        return True
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    """SCL 译码器（基于 Permuted SC 结构）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _frozen_set(frozen_bits)
        frozen_arr = np.asarray(frozen_bits)
        if frozen_arr.dtype != bool:
            frozen_arr = frozen_arr.astype(int) == 1
        self.info_indices = np.where(~frozen_arr)[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _path_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = reorder_llr_for_decode(llr_ch)
        N, n = self.N, self.n

        paths = [
            {
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.full((N, n + 1), np.nan),
                "pm": 0.0,
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []

            for path in paths:
                L = path["L"]
                B = path["B"]
                self._update_llrs(L, B, l)
                llr_val = L[l, n]

                if l in self.frozen:
                    new_L = L.copy()
                    new_B = B.copy()
                    new_B[l, n] = 0
                    self._update_bits(new_B, l)
                    candidates.append(
                        {"L": new_L, "B": new_B, "pm": path["pm"] + self._path_penalty(llr_val, 0)}
                    )
                else:
                    for u_bit in (0, 1):
                        new_L = L.copy()
                        new_B = B.copy()
                        new_B[l, n] = u_bit
                        self._update_bits(new_B, l)
                        candidates.append(
                            {
                                "L": new_L,
                                "B": new_B,
                                "pm": path["pm"] + self._path_penalty(llr_val, u_bit),
                            }
                        )

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path["B"][:, n].astype(int)[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        u_hat = best["B"][:, n].astype(int)
        return u_hat, best["pm"]
