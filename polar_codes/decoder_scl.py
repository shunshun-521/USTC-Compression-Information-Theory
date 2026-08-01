"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
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
    """检验 CRC 校验位是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits[:-crc_length], poly, crc_length)
    expected = bits[-crc_length:]
    actual = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.array_equal(expected, actual)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)

    def _compute_llr(self, l, L, B):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s],
                        B[j - branch_size, s + 1],
                    )
        return L[l, self.n]

    def _update_bits(self, l, B):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L0 = np.zeros((N, n + 1), dtype=np.float64)
        L0[:, 0] = llr_ch[self.rev]

        paths = [{'pm': 0.0,
                  'L': L0.copy(),
                  'B': np.zeros((N, n + 1), dtype=np.int32),
                  'u_hat': np.zeros(N, dtype=int)}]

        for phi in range(N):
            l = _bit_reversed(phi, n)
            candidates = []

            for path in paths:
                llr = self._compute_llr(l, path['L'], path['B'])

                if self.frozen_bits[l]:
                    candidates.append(
                        (path['pm'] + self._pm_penalty(llr, 0), path, 0)
                    )
                else:
                    for u in (0, 1):
                        candidates.append(
                            (path['pm'] + self._pm_penalty(llr, u), path, u)
                        )

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for new_pm, old_path, u in candidates:
                new_path = {
                    'pm': new_pm,
                    'L': old_path['L'].copy(),
                    'B': old_path['B'].copy(),
                    'u_hat': old_path['u_hat'].copy(),
                }
                new_path['u_hat'][l] = u
                new_path['B'][l, n] = u
                self._update_bits(l, new_path['B'])
                new_paths.append(new_path)

            paths = new_paths

        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            valid = [p for p in paths if crc_check(p['u_hat'][info_mask],
                                                    self.crc_length)]
            best = min(valid or paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].copy(), best['pm']
