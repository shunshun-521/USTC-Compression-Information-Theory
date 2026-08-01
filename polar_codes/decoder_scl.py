"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << (crc_length + 1)) - 1)
        if reg & (1 << crc_length):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & ((1 << (crc_length + 1)) - 1)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_penalty(self, llr, u):
        expected = 0 if llr >= 0 else 1
        return 0.0 if u == expected else abs(llr)

    def _compute_llr(self, L, B, phi):
        l = _bit_reversed(phi, self.n)
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 1 << (s + 1)
            half = block // 2
            for j in range(l, self.N, block):
                if j % block < half:
                    L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
                else:
                    top_bit = int(B[j - half, s + 1])
                    L[j, s + 1] = g_operation(L[j - half, s], L[j, s], top_bit)

    def _propagate_bits(self, B, phi):
        l = _bit_reversed(phi, self.n)
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 1 << s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    B[j - half, s - 1] = int(B[j, s]) ^ int(B[j - half, s])
                    B[j, s - 1] = int(B[j, s])

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [{
            'L': np.zeros((self.N, self.n + 1), dtype=np.float64),
            'B': np.zeros((self.N, self.n + 1), dtype=int),
            'pm': 0.0,
            'u_hat': np.zeros(self.N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                self._compute_llr(path['L'], path['B'], phi)
                llr0 = path['L'][l, self.n]

                if l in self.frozen_set:
                    u = 0
                    new_path = {
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'pm': path['pm'] + self._path_penalty(llr0, u),
                        'u_hat': path['u_hat'].copy(),
                    }
                    new_path['u_hat'][phi] = u
                    new_path['B'][l, self.n] = u
                    self._propagate_bits(new_path['B'], phi)
                    new_paths.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': path['pm'] + self._path_penalty(llr0, u),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new_path['u_hat'][phi] = u
                        new_path['B'][l, self.n] = u
                        self._propagate_bits(new_path['B'], phi)
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['u_hat'], self.crc_length)]
            best = min(valid, key=lambda p: p['pm']) if valid else min(paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['B'][:, self.n].astype(int), best['pm']
