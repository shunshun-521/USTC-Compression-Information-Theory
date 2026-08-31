"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Permuted SC
"""
import numpy as np
import math
import copy

from encoder import bit_reversed_index
from decoder_sc import (
    f_operation, g_operation, _active_llr_level, _active_bit_level,
    _update_llrs, _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    width = crc_length
    top = 1 << (width - 1)
    mask = (1 << width) - 1
    for bit in info_bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (width - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch
        return {'L': L, 'B': B, 'pm': 0.0, 'u_hat': np.zeros(self.N, dtype=int)}

    def _copy_path(self, path):
        return {
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'pm': path['pm'],
            'u_hat': path['u_hat'].copy(),
        }

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for step, l in enumerate(self.decode_order):
            candidates = []
            for path in paths:
                _update_llrs(path['L'], path['B'], l, self.n)
                llr = path['L'][l, self.n]

                if l in self.frozen_set:
                    penalty = 0.0 if llr >= 0 else abs(llr)
                    new = self._copy_path(path)
                    new['pm'] += penalty
                    new['B'][l, self.n] = 0
                    new['u_hat'][l] = 0
                    _update_bits(new['B'], l, self.n, self.N)
                    candidates.append(new)
                else:
                    for bit in (0, 1):
                        hard = 0 if llr >= 0 else 1
                        penalty = 0.0 if bit == hard else abs(llr)
                        new = self._copy_path(path)
                        new['pm'] += penalty
                        new['B'][l, self.n] = bit
                        new['u_hat'][l] = bit
                        _update_bits(new['B'], l, self.n, self.N)
                        candidates.append(new)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info = p['u_hat'][self.info_indices]
                if crc_check(info, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['u_hat'].copy(), best['pm']
