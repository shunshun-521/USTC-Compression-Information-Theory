"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    f_operation, g_operation, _active_llr_level, _active_bit_level
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_division(bits, poly, crc_len):
    reg = np.zeros(crc_len, dtype=np.int8)
    for bit in bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            for j in range(crc_len):
                if (poly >> (crc_len - j)) & 1:
                    reg[j] ^= feedback
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_division(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_division(bits, poly, crc_length)
    return np.all(remainder == 0)


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy 优化）。
    frozen_bits: True/1 表示冻结位
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _update_llrs(self, L, B, l):
        active = _active_llr_level(l, self.n)
        for s in range(self.n - active, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _propagate_bits(self, B, l):
        if l < self.N // 2:
            return
        active_bit = _active_bit_level(l, self.n)
        for s in range(self.n, self.n - active_bit, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [{
            'pm': 0.0,
            'L': np.zeros((self.N, self.n + 1)),
            'B': np.zeros((self.N, self.n + 1), dtype=np.int8),
            'u_hat': np.zeros(self.N, dtype=np.int8),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path['L'], path['B'], l)
                llr = path['L'][l, self.n]

                if self.frozen_bits[l]:
                    new_path = {
                        'pm': path['pm'] + self._pm_penalty(llr, 0),
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'u_hat': path['u_hat'].copy(),
                    }
                    new_path['B'][l, self.n] = 0
                    new_path['u_hat'][l] = 0
                    self._propagate_bits(new_path['B'], l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = {
                            'pm': path['pm'] + self._pm_penalty(llr, bit),
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new_path['B'][l, self.n] = bit
                        new_path['u_hat'][l] = bit
                        self._propagate_bits(new_path['B'], l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

        crc_pass = []
        for p in paths:
            if self.crc_length > 0:
                info_bits = p['u_hat'][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            else:
                crc_pass.append(p)

        best = min(crc_pass if crc_pass else paths, key=lambda p: p['pm'])
        return best['u_hat'].copy(), best['pm']
