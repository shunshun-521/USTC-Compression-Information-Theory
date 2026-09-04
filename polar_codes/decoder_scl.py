"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import copy
import numpy as np
from decoder_sc import sc_decode, _map_channel_llr
from decoder_sc_core import (
    SCD, upper_llr, lower_llr, hard_decision,
    bit_reversed, active_llr_level, active_bit_level,
)


CRC8_POLY = 0xE0
CRC16_POLY = 0xA001


def _crc_mod(bits, poly, crc_length):
    crc = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        feedback = ((crc >> (crc_length - 1)) ^ int(bit)) & 1
        crc = (crc << 1) & mask
        if feedback:
            crc ^= poly
    return crc


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_mod(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)]),
        poly, crc_length
    )
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_mod(bits, poly, crc_length) == 0


class _SCLPath:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n, likelihoods):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = likelihoods
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        p = _SCLPath.__new__(_SCLPath)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


def _update_llrs_path(path, l, n, N):
    for s in range(n - active_llr_level(l, n), n):
        block_size = int(2 ** (s + 1))
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
            else:
                top_bit = path.B[j - branch_size, s + 1]
                path.L[j, s + 1] = lower_llr(path.L[j, s], path.L[j - branch_size, s], top_bit)


def _update_bits_path(path, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = int(2 ** s)
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                path.B[j, s - 1] = path.B[j, s]


def _pm_update(pm, llr_val, u_bit):
    llr_sign_bit = 0 if llr_val >= 0 else 1
    if u_bit != llr_sign_bit:
        pm += abs(llr_val)
    return pm


class SCLDecoder:
    """SCL 译码器（Vangala 置换 SC 核心 + 路径列表）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        mapped = _map_channel_llr(llr_ch)
        paths = [_SCLPath(self.N, self.n, mapped)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs_path(path, l, self.n, self.N)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = path.copy()
                    new_path.pm = _pm_update(path.pm, llr_val, 0)
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    _update_bits_path(new_path, l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = path.copy()
                        new_path.pm = _pm_update(path.pm, llr_val, u_bit)
                        new_path.B[l, self.n] = u_bit
                        new_path.u_hat[l] = u_bit
                        _update_bits_path(new_path, l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
