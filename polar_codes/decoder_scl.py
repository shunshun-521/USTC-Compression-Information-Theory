"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
)


# CRC-8: 0x07, CRC-16: 0x8005
_CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= (bit << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{
            'L': np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            'B': np.full((self.N, self.n + 1), np.nan),
            'pm': 0.0,
            'u': np.zeros(self.N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for phi in [_bit_reversed(i, self.n) for i in range(self.N)]:
            new_paths = []
            for path in paths:
                self._update_llrs(path, phi)
                llr = path['L'][phi, self.n]
                if np.isnan(llr):
                    llr = 0.0

                if self.frozen_bits[phi]:
                    penalty = 0.0 if llr >= 0 else abs(llr)
                    path['pm'] += penalty
                    path['B'][phi, self.n] = 0
                    path['u'][phi] = 0
                    self._update_bits(path, phi)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p = self._lazy_copy(path)
                        if bit == 0:
                            penalty = 0.0 if llr >= 0 else abs(llr)
                        else:
                            penalty = 0.0 if llr < 0 else abs(llr)
                        p['pm'] += penalty
                        p['B'][phi, self.n] = bit
                        p['u'][phi] = bit
                        self._update_bits(p, phi)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_valid(p['u'])]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['u'].copy(), best['pm']

    def _crc_valid(self, u_hat):
        info_bits = u_hat[self.info_indices]
        if len(info_bits) < self.crc_length:
            return False
        return crc_check(info_bits, self.crc_length)

    def _lazy_copy(self, path):
        return {
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'pm': path['pm'],
            'u': path['u'].copy(),
        }

    def _update_llrs(self, path, l):
        L = path['L']
        B = path['B']
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], int(top_bit))

    def _update_bits(self, path, l):
        B = path['B']
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]
