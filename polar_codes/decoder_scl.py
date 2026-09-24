"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _f_boxplus, _lower_llr, _bit_reversed,
    _active_llr_level, _active_bit_level, g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_divide(bits, poly, crc_length):
    reg = [0] * crc_length
    for bit in bits:
        feedback = bit ^ reg[0]
        reg = reg[1:] + [0]
        if feedback:
            poly_bits = [(poly >> i) & 1 for i in range(crc_length - 1, -1, -1)]
            for i in range(crc_length):
                if poly_bits[i]:
                    reg[i] ^= feedback
    return np.array(reg, dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_divide(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_divide(bits, poly, crc_length)
    return np.all(remainder == 0)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.br = bit_reversal_permutation(N)
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(self.frozen_bits == 0)[0]

    def _init_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_ch[self.br]
        return {'L': L, 'B': B, 'pm': 0.0, 'u_hat': np.zeros(self.N, dtype=np.int8)}

    def _copy_path(self, path):
        return {
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'pm': path['pm'],
            'u_hat': path['u_hat'].copy(),
        }

    def _update_llrs(self, path, l):
        L = path['L']
        B = path['B']
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _f_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s],
                        int(B[j - branch_size, s + 1])
                    )

    def _update_bits(self, path, l):
        B = path['B']
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        paths = [self._init_path(llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path['L'][l, self.n]

                if l in self.frozen:
                    new_path = self._copy_path(path)
                    penalty = 0.0 if llr >= 0 else abs(llr)
                    new_path['pm'] += penalty
                    new_path['u_hat'][l] = 0
                    new_path['B'][l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        expected = 0 if llr >= 0 else 1
                        penalty = 0.0 if bit == expected else abs(llr)
                        new_path['pm'] += penalty
                        new_path['u_hat'][l] = bit
                        new_path['B'][l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p['u_hat'][self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].copy(), best['pm']
