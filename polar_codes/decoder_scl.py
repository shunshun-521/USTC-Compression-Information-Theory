"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
基于 Permuted SCD（Vangala et al., 2014）
"""
import math
import numpy as np
from decoder_sc import (
    SCDecoder, bit_reversed, active_llr_level, active_bit_level,
    f_operation, g_operation, bit_reversal_permutation,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
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
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器（Permuted SCD + 路径列表）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr = llr_ch[br]

        paths = [{
            'pm': 0.0,
            'L': np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            'B': np.full((self.N, self.n + 1), np.nan),
        }]
        paths[0]['L'][:, 0] = llr

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                L_work = path['L'].copy()
                B_work = path['B'].copy()
                self._update_llrs(L_work, B_work, l)
                llr_val = L_work[l, self.n]

                u_opts = [0] if l in self.frozen_set else [0, 1]
                for u_val in u_opts:
                    Lc = L_work.copy()
                    Bc = B_work.copy()
                    pm = path['pm']
                    hard = 0 if llr_val >= 0 else 1
                    if u_val != hard:
                        pm += abs(llr_val)
                    Bc[l, self.n] = u_val
                    self._update_bits(Bc, l)
                    candidates.append({'pm': pm, 'L': Lc, 'B': Bc})

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_hat = p['B'][:, self.n].astype(int)
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        u_hat = best['B'][:, self.n].astype(int)
        return u_hat, best['pm']
