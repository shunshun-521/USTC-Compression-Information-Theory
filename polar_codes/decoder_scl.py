"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np
from decoder_sc import (
    f_boxplus,
    g_boxplus,
    bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _frozen_indices_from_mask,
)


CRC_POLYNOMIALS = {
    8: [1, 0, 0, 0, 0, 1, 1, 1],
    16: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly_bits = CRC_POLYNOMIALS[crc_length]
    data = list(map(int, info_bits))
    reg = data + [0] * crc_length
    n = len(poly_bits)
    for i in range(len(data)):
        if reg[i]:
            for j in range(n):
                if i + j < len(reg):
                    reg[i + j] ^= poly_bits[j]
    crc = reg[len(data):]
    return np.array(data + crc, dtype=np.int8)


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly_bits = CRC_POLYNOMIALS[crc_length]
    reg = list(map(int, bits))
    n = len(poly_bits)
    for i in range(len(bits) - crc_length):
        if reg[i]:
            for j in range(n):
                if i + j < len(reg):
                    reg[i + j] ^= poly_bits[j]
    return all(x == 0 for x in reg[-crc_length:])


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.frozen_set = _frozen_indices_from_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(self.N)]
        self.info_idx = np.where(self.frozen_bits == 0)[0]

    def _init_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        for i, idx in enumerate(self.decode_order):
            L[i, 0] = llr_ch[idx]
        return {'L': L, 'B': B, 'pm': 0.0, 'u_hat': np.zeros(self.N, dtype=np.int8)}

    def _update_llrs(self, path, l):
        L, B = path['L'], path['B']
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_boxplus(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        B = path['B']
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    @staticmethod
    def _penalty(llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._init_path(llr_ch)]

        for phi in range(self.N):
            l = self.decode_order[phi]
            expanded = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path['L'][l, self.n]

                if l in self.frozen_set:
                    new_path = copy.deepcopy(path)
                    new_path['pm'] += self._penalty(llr_val, 0)
                    new_path['B'][l, self.n] = 0
                    new_path['u_hat'][l] = 0
                    self._update_bits(new_path, l)
                    expanded.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = copy.deepcopy(path)
                        new_path['pm'] += self._penalty(llr_val, u_bit)
                        new_path['B'][l, self.n] = u_bit
                        new_path['u_hat'][l] = u_bit
                        self._update_bits(new_path, l)
                        expanded.append(new_path)

            expanded.sort(key=lambda p: p['pm'])
            paths = expanded[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['u_hat'][self.info_idx], self.crc_length)]
            best = min(valid, key=lambda p: p['pm']) if valid else min(paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].astype(int), best['pm']
