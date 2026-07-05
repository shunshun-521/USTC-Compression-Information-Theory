"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    active_llr_level,
    active_bit_level,
)
from encoder import bit_reversed


CRC8_POLY_BITS = [1, 0, 0, 0, 0, 0, 1, 1, 1]
CRC16_POLY_BITS = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _gf2_crc_remainder(data, poly_bits):
    msg = [int(b) for b in data]
    divisor = poly_bits
    deg = len(divisor) - 1
    for i in range(len(msg) - deg):
        if msg[i]:
            for j in range(len(divisor)):
                msg[i + j] ^= divisor[j]
    return np.array(msg[-deg:], dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY_BITS if crc_length == 8 else CRC16_POLY_BITS
    extended = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    remainder = _gf2_crc_remainder(extended, poly)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY_BITS if crc_length == 8 else CRC16_POLY_BITS
    remainder = _gf2_crc_remainder(bits, poly)
    return np.all(remainder == 0)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        L[:, 0] = llr_ch
        B[:, 0] = 0.0
        return {'pm': 0.0, 'L': L, 'B': B}

    def _copy_path(self, path):
        return {
            'pm': path['pm'],
            'L': path['L'].copy(),
            'B': path['B'].copy(),
        }

    def _update_llrs(self, path, l):
        L = path['L']
        B = path['B']
        n = self.n
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

    def _update_bits(self, path, l, bit):
        B = path['B']
        n = self.n
        B[l, n] = bit
        if l < self.N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path['L'][l, self.n]

                if self.frozen_bits[l]:
                    new_path = self._copy_path(path)
                    new_path['pm'] += self._penalty(llr, 0)
                    self._update_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path['pm'] += self._penalty(llr, bit)
                        self._update_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['B'][:, self.n], self.crc_length)]
            best = min(valid, key=lambda p: p['pm']) if valid else min(paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['B'][:, self.n].astype(np.int8), best['pm']
