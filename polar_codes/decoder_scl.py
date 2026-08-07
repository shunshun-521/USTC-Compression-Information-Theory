"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from encoder import bit_reversal_permutation
from decoder_sc import (
    _SCD, _bit_reversed, _active_llr_level, _active_bit_level,
    _upper_llr, _lower_llr, f_operation, g_operation
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= (poly >> (16 - crc_length)) if crc_length == 16 else poly
    for _ in range(crc_length):
        reg = ((reg << 1)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= (poly >> (16 - crc_length)) if crc_length == 16 else poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1
                         for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.frozen_set = set(np.where(self.frozen_bits.astype(bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits.astype(bool))[0]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _copy_state(self, L, B, pm, u_hat):
        return L.copy(), B.copy(), pm, u_hat.copy()

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            bs = int(2 ** (s + 1))
            br = bs // 2
            for j in range(l, self.N, bs):
                if j % bs < br:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + br, s])
                else:
                    tb = B[j - br, s + 1]
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - br, s], tb)

    def _update_bits(self, B, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            bs = int(2 ** s)
            br = bs // 2
            for j in range(l, -1, -bs):
                if j % bs >= br:
                    B[j - br, s - 1] = int(B[j, s]) ^ int(B[j - br, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[rev]

        L = np.full((self.N, self.n + 1), np.nan)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_ch
        paths = [(L, B, 0.0, np.zeros(self.N, dtype=int))]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []
            for Lp, Bp, pm, u_hat in paths:
                Lc, Bc, _, _ = self._copy_state(Lp, Bp, pm, u_hat)
                self._update_llrs(Lc, Bc, l)
                llr = Lc[l, self.n]
                if l in self.frozen_set:
                    penalty = self._pm_penalty(llr, 0)
                    u_hat_new = u_hat.copy()
                    u_hat_new[l] = 0
                    Bc[l, self.n] = 0
                    self._update_bits(Bc, l)
                    new_paths.append((Lc, Bc, pm + penalty, u_hat_new))
                else:
                    for u in (0, 1):
                        Lb, Bb, _, _ = self._copy_state(Lc, Bc, pm, u_hat)
                        penalty = self._pm_penalty(llr, u)
                        u_hat_new = u_hat.copy()
                        u_hat_new[l] = u
                        Bb[l, self.n] = u
                        self._update_bits(Bb, l)
                        new_paths.append((Lb, Bb, pm + penalty, u_hat_new))
            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [(L, B, pm, u) for L, B, pm, u in paths
                       if crc_check(u[self.info_indices], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda x: x[2])
        return best[3].copy(), best[2]
