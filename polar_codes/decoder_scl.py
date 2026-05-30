"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_index
from decoder_sc import (
    upper_llr,
    lower_llr,
    _active_llr_level,
    _active_bit_level,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_len):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_len - 1)
        for _ in range(8 if crc_len <= 8 else 16):
            if crc_len <= 8:
                msb = (reg >> 7) & 1
                reg = ((reg << 1) & 0xFF) ^ (poly if msb else 0)
            else:
                msb = (reg >> 15) & 1
                reg = ((reg << 1) & 0xFFFF) ^ (poly if msb else 0)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _pm_penalty(llr_val, u_bit):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr_val)


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], top_bit)


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.L = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = []
        L0 = np.full((N, n + 1), np.nan, dtype=np.float64)
        B0 = np.zeros((N, n + 1), dtype=np.int8)
        L0[:, 0] = llr_ch
        paths.append((0.0, L0, B0))

        for l in [bit_reversal_index(i, n) for i in range(N)]:
            new_paths = []
            for pm, L, B in paths:
                _update_llrs(L, B, l, n, N)
                cur_llr = L[l, n]
                if l in self.frozen_set:
                    pen = _pm_penalty(cur_llr, 0)
                    B2 = B.copy()
                    L2 = L.copy()
                    B2[l, n] = 0
                    _update_bits(B2, l, n, N)
                    new_paths.append((pm + pen, L2, B2))
                else:
                    for u_bit in (0, 1):
                        L2 = L.copy()
                        B2 = B.copy()
                        B2[l, n] = u_bit
                        _update_bits(B2, l, n, N)
                        new_paths.append((pm + _pm_penalty(cur_llr, u_bit), L2, B2))
            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.L]

        if self.crc_length > 0:
            valid = []
            for pm, L, B in paths:
                u = B[:, n].astype(int)
                if crc_check(u, self.crc_length):
                    valid.append((pm, L, B))
            if valid:
                paths = valid

        best_pm, _, best_B = min(paths, key=lambda x: x[0])
        return best_B[:, n].astype(int), best_pm
