"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation, bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """MSB-first CRC remainder（每位一次移位）。"""
    reg = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self._rev = bit_reversal_permutation(N)

    @staticmethod
    def _path_metric_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _clone_state(self, L, B, pm, u_hat):
        return L.copy(), B.copy(), pm, u_hat.copy()

    def _update_llrs(self, L, B, l):
        n = self.n
        for s in range(n - active_llr_level(l, n), n):
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
        n = self.n
        if l < self.N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        llr_internal = llr_ch[self._rev]
        L0 = np.zeros((N, n + 1), dtype=np.float64)
        B0 = np.zeros((N, n + 1), dtype=np.int8)
        L0[:, 0] = llr_internal

        paths = [(L0, B0, 0.0, np.zeros(N, dtype=int))]

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []

            for L, B, pm, u_hat in paths:
                self._update_llrs(L, B, l)
                llr = L[l, n]

                if self.frozen_bits[l]:
                    penalty = self._path_metric_penalty(llr, 0)
                    u_hat[l] = 0
                    B[l, n] = 0
                    self._update_bits(B, l)
                    candidates.append((L, B, pm + penalty, u_hat))
                else:
                    for bit in (0, 1):
                        Lc, Bc, _, ucopy = self._clone_state(L, B, pm, u_hat)
                        penalty = self._path_metric_penalty(llr, bit)
                        ucopy[l] = bit
                        Bc[l, n] = bit
                        self._update_bits(Bc, l)
                        candidates.append((Lc, Bc, pm + penalty, ucopy))

            candidates.sort(key=lambda x: x[2])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_pass = []
            for L, B, pm, u_hat in paths:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append((L, B, pm, u_hat))
            pool = crc_pass if crc_pass else paths
        else:
            pool = paths

        best = min(pool, key=lambda x: x[2])
        return best[3].copy(), best[2]
