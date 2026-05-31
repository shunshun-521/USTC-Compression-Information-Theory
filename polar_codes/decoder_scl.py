"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    bit_reversed_index,
    active_llr_level,
    active_bit_level,
    upper_llr_boxplus,
    lower_llr_boxplus,
    f_operation,
    g_operation,
)

# CRC-8: x^8 + x^2 + x + 1 (0x07)
_CRC8_POLY = 0x07
# CRC-16: 0x8005
_CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    return _CRC8_POLY if crc_length == 8 else _CRC16_POLY


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy 路径管理）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.info_indices = np.array(
            [i for i in range(N) if i not in self.frozen], dtype=int
        )

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.list_size

        paths = [{
            'pm': 0.0,
            'L': np.zeros((self.N, self.n + 1), dtype=np.float64),
            'B': np.zeros((self.N, self.n + 1), dtype=int),
            'u': np.zeros(self.N, dtype=int),
            'active': True,
        }]
        paths[0]['L'][:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            new_paths = []

            for p in paths:
                if not p['active']:
                    continue
                self._update_llrs_path(p, l)
                cur_llr = p['L'][l, self.n]

                if l in self.frozen:
                    pen = 0.0 if cur_llr >= 0 else abs(cur_llr)
                    p2 = self._clone_path(p)
                    p2['pm'] += pen
                    p2['B'][l, self.n] = 0
                    p2['u'][l] = 0
                    self._update_bits_path(p2, l)
                    new_paths.append(p2)
                else:
                    for bit in (0, 1):
                        p2 = self._clone_path(p)
                        pen = 0.0 if (bit == 0 and cur_llr >= 0) or (bit == 1 and cur_llr < 0) else abs(cur_llr)
                        p2['pm'] += pen
                        p2['B'][l, self.n] = bit
                        p2['u'][l] = bit
                        self._update_bits_path(p2, l)
                        new_paths.append(p2)

            new_paths.sort(key=lambda x: x['pm'])
            paths = new_paths[:L]
            if not paths:
                paths = new_paths[:1]

        best = self._select_path(paths)
        return best['u'].copy(), best['pm']

    def _select_path(self, paths):
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p['u'][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                return min(valid, key=lambda x: x['pm'])
        return min(paths, key=lambda x: x['pm'])

    @staticmethod
    def _clone_path(p):
        return {
            'pm': p['pm'],
            'L': p['L'].copy(),
            'B': p['B'].copy(),
            'u': p['u'].copy(),
            'active': True,
        }

    def _update_llrs_path(self, p, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    p['L'][j, s + 1] = upper_llr_boxplus(p['L'][j, s], p['L'][j + branch_size, s])
                else:
                    top_bit = p['B'][j - branch_size, s + 1]
                    p['L'][j, s + 1] = lower_llr_boxplus(
                        p['L'][j, s], p['L'][j - branch_size, s], top_bit
                    )

    def _update_bits_path(self, p, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    p['B'][j - branch_size, s - 1] = (
                        int(p['B'][j, s]) ^ int(p['B'][j - branch_size, s])
                    )
                    p['B'][j, s - 1] = p['B'][j, s]


def scl_decode_equivalent_sc(llr, frozen_bits):
    """L=1 的 SCL 应等价于 SC。"""
    dec = SCLDecoder(len(llr), frozen_bits, list_size=1)
    u, _ = dec.decode(llr)
    return u
