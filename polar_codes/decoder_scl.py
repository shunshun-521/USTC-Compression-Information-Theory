"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import sc_decode
from encoder import bit_reversal_permutation
from scd_core import (
    SCDecoder,
    active_bit_level,
    active_llr_level,
    bit_reversed,
    hard_decision,
    lower_llr,
    upper_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    msg_int = 0
    for b in info_bits:
        msg_int = (msg_int << 1) | int(b)
    msg_int <<= crc_length
    order = len(info_bits) + crc_length
    for i in range(order - 1, crc_length - 1, -1):
        if (msg_int >> i) & 1:
            msg_int ^= poly << (i - crc_length)
    crc_val = msg_int & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(crc_val >> (crc_length - 1 - j)) & 1 for j in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    msg_int = 0
    for b in bits:
        msg_int = (msg_int << 1) | int(b)
    for i in range(len(bits) - 1, crc_length - 1, -1):
        if (msg_int >> i) & 1:
            msg_int ^= poly << (i - crc_length)
    return (msg_int & ((1 << crc_length) - 1)) == 0


class PathState:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = PathState(len(self.u_hat), int(math.log2(len(self.u_hat))), self.L[:, 0])
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.br = bit_reversal_permutation(N)

    def _permute_llr(self, llr_ch):
        llr = np.zeros(self.N, dtype=np.float64)
        llr[self.br] = llr_ch
        return llr

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    btm = path.L[j, s]
                    top = path.L[j - branch_size, s]
                    bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = lower_llr(btm, top, bit)

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = int(2 ** s)
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr = self._permute_llr(llr_ch)
        paths = [PathState(self.N, self.n, llr)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    penalty = abs(llr_val) if llr_val < 0 else 0.0
                    new_path = path.copy()
                    new_path.pm += penalty
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        consistent = (bit == 0 and llr_val >= 0) or (bit == 1 and llr_val < 0)
                        if not consistent:
                            new_path.pm += abs(llr_val)
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
