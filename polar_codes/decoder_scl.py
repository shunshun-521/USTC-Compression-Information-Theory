"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    clip_llr,
    f_operation,
    g_operation,
)
from encoder import bit_reversed_index

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_ch
        return {
            'pm': 0.0,
            'u_hat': np.zeros(self.N, dtype=int),
            'L': L,
            'B': B,
        }

    def _copy_path(self, path):
        return {
            'pm': path['pm'],
            'u_hat': path['u_hat'].copy(),
            'L': path['L'].copy(),
            'B': path['B'].copy(),
        }

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 1 << (s + 1)
            half = block // 2
            for j in range(l, self.N, block):
                if j % block < half:
                    path['L'][j, s + 1] = f_operation(path['L'][j, s], path['L'][j + half, s])
                else:
                    path['L'][j, s + 1] = g_operation(
                        path['L'][j, s],
                        path['L'][j - half, s],
                        path['B'][j - half, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 1 << s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    path['B'][j - half, s - 1] = int(path['B'][j, s]) ^ int(path['B'][j - half, s])
                    path['B'][j, s - 1] = path['B'][j, s]

    def decode(self, llr_ch):
        llr_ch = clip_llr(llr_ch)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path['L'][l, self.n]

                if l in self.frozen_set:
                    child = self._copy_path(path)
                    if llr < 0:
                        child['pm'] += abs(llr)
                    child['u_hat'][l] = 0
                    child['B'][l, self.n] = 0
                    self._update_bits(child, l)
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        child = self._copy_path(path)
                        if (bit == 0 and llr < 0) or (bit == 1 and llr >= 0):
                            child['pm'] += abs(llr)
                        child['u_hat'][l] = bit
                        child['B'][l, self.n] = bit
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['u_hat'], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['u_hat'], best['pm']
