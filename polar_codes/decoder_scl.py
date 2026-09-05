"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversed, bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    active_llr_level,
    active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    """对比特序列做 CRC 除法，返回余数"""
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_process(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否包含正确的 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_process(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_positions = np.where(~self.frozen_bits)[0]
        self.list_size = list_size
        self.crc_length = crc_length

        if list_size == 1 and crc_length == 0:
            from decoder_sc import sc_decode
            self._sc_decode = sc_decode
            self._use_sc = True
        else:
            self._use_sc = False

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self._use_sc:
            u_hat = self._sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[rev]

        paths = []
        L0 = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B0 = np.zeros((self.N, self.n + 1), dtype=int)
        L0[:, 0] = llr_ch
        paths.append({'L': L0, 'B': B0, 'pm': 0.0, 'u_hat': np.zeros(self.N, dtype=int)})

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path['L'], path['B'], l)
                llr = path['L'][l, self.n]

                if l in self.frozen_set:
                    new_path = {
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'pm': path['pm'] + self._path_metric_penalty(llr, 0),
                        'u_hat': path['u_hat'].copy(),
                    }
                    new_path['B'][l, self.n] = 0
                    new_path['u_hat'][l] = 0
                    self._update_bits(new_path['B'], l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': path['pm'] + self._path_metric_penalty(llr, bit),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new_path['B'][l, self.n] = bit
                        new_path['u_hat'][l] = bit
                        self._update_bits(new_path['B'], l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p['u_hat'][self.info_positions], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].copy(), best['pm']
