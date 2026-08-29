"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
)
from encoder import bit_reversed


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << (crc_length + 1)) - 1)
        if reg & (1 << crc_length):
            reg ^= poly
    for _ in range(crc_length):
        reg <<= 1
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


def _pm_update(pm, llr, u_bit):
    hard = 0 if llr >= 0 else 1
    if u_bit != hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{
            'pm': 0.0,
            'L': np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            'B': np.full((self.N, self.n + 1), np.nan),
            'u_hat': np.zeros(self.N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path['L'], path['B'], l)
                llr = path['L'][l, self.n]

                if self.frozen_bits[l]:
                    new_path = {
                        'pm': _pm_update(path['pm'], llr, 0),
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'u_hat': path['u_hat'].copy(),
                    }
                    new_path['B'][l, self.n] = 0
                    new_path['u_hat'][l] = 0
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = {
                            'pm': _pm_update(path['pm'], llr, u_bit),
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new_path['B'][l, self.n] = u_bit
                        new_path['u_hat'][l] = u_bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

            for path in paths:
                self._update_bits(path['B'], l)

        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            valid = [
                p for p in paths
                if crc_check(p['u_hat'][info_positions], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['u_hat'].copy(), best['pm']
