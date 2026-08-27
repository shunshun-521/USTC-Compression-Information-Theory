"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
基于 Vangala 置换 SC 结构
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    _active_llr_level, _active_bit_level,
    _upper_llr_exact, _lower_llr_exact,
    f_operation, g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc8_run(bits):
    """CRC-8 (poly 0x07) 逐比特运算"""
    crc = 0
    for bit in bits:
        msb = (crc >> 7) & 1
        crc = (crc << 1) & 0xFF
        if msb ^ int(bit):
            crc ^= CRC8_POLY
    return crc


def _crc16_run(bits):
    """CRC-16 (poly 0x8005) 逐比特运算"""
    crc = 0
    for bit in bits:
        msb = (crc >> 15) & 1
        crc = (crc << 1) & 0xFFFF
        if msb ^ int(bit):
            crc ^= CRC16_POLY
    return crc


def _crc_run(bits, crc_length):
    return _crc8_run(bits) if crc_length == 8 else _crc16_run(bits)


def _int_to_bits(val, n):
    return np.array([(val >> (n - 1 - i)) & 1 for i in range(n)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    crc_val = _crc_run(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), crc_length
    )
    crc_bits = _int_to_bits(crc_val, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    crc_part = bits[-crc_length:]
    expected = _crc_run(
        np.concatenate([info, np.zeros(crc_length, dtype=int)]), crc_length
    )
    actual = sum(int(crc_part[i]) << (crc_length - 1 - i) for i in range(crc_length))
    return expected == actual


class SCLDecoder:
    """SCL 译码器（Lazy Copy + Vangala 置换顺序）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, use_min_sum=False):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.use_min_sum = use_min_sum
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.upper = f_operation if use_min_sum else _upper_llr_exact
        self.lower = _lower_llr_exact if not use_min_sum else None

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_ch
        return {'L': L, 'B': B, 'pm': 0.0, 'u_hat': np.zeros(self.N, dtype=int)}

    def _copy_path(self, path):
        return {
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'pm': path['pm'],
            'u_hat': path['u_hat'].copy(),
        }

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path['L'][j, s + 1] = self.upper(path['L'][j, s], path['L'][j + branch_size, s])
                else:
                    top_bit = int(path['B'][j - branch_size, s + 1])
                    if self.use_min_sum:
                        path['L'][j, s + 1] = g_operation(
                            path['L'][j - branch_size, s], path['L'][j, s], top_bit
                        )
                    else:
                        path['L'][j, s + 1] = self.lower(
                            path['L'][j, s], path['L'][j - branch_size, s], top_bit
                        )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path['B'][j - branch_size, s - 1] = (
                        int(path['B'][j, s]) ^ int(path['B'][j - branch_size, s])
                    )
                    path['B'][j, s - 1] = path['B'][j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            for path in paths:
                self._update_llrs(path, l)

            if l in self.frozen_set:
                for path in paths:
                    path['pm'] += self._pm_penalty(path['L'][l, self.n], 0)
                    path['u_hat'][l] = 0
                    path['B'][l, self.n] = 0
            else:
                candidates = []
                for path in paths:
                    llr = path['L'][l, self.n]
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path['pm'] += self._pm_penalty(llr, bit)
                        new_path['u_hat'][l] = bit
                        new_path['B'][l, self.n] = bit
                        candidates.append(new_path)
                candidates.sort(key=lambda p: p['pm'])
                paths = candidates[:self.list_size]

            for path in paths:
                self._update_bits(path, l)

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p['u_hat'][self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['u_hat'].copy(), best['pm']
