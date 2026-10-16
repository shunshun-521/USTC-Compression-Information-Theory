"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    lower_llr,
    sc_decode,
    upper_llr,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, crc_length, poly):
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    rem = _crc_process(padded, crc_length, poly)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_process(bits, crc_length, poly) == 0


class SCLDecoder:
    """SCL 译码器（路径复制 + log-domain SC 核）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_ch
        return {"L": L, "B": B, "pm": 0.0}

    def _update_llrs(self, path, l):
        L = path["L"]
        B = path["B"]
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], int(top_bit)
                    )

    def _update_bits(self, path, l):
        B = path["B"]
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [self._new_path(llr_ch)]
        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path["L"][l, self.n]
                if l in self.frozen_set:
                    p = copy.deepcopy(path)
                    p["pm"] += self._pm_penalty(llr_bit, 0)
                    p["B"][l, self.n] = 0
                    self._update_bits(p, l)
                    candidates.append(p)
                else:
                    for u in (0, 1):
                        p = copy.deepcopy(path)
                        p["pm"] += self._pm_penalty(llr_bit, u)
                        p["B"][l, self.n] = u
                        self._update_bits(p, l)
                        candidates.append(p)
            candidates.sort(key=lambda x: x["pm"])
            paths = candidates[: self.list_size]

        paths.sort(key=lambda x: x["pm"])
        if self.crc_length > 0:
            for p in paths:
                u_hat = p["B"][:, self.n].astype(np.int8)
                info = u_hat[self.info_positions]
                if crc_check(info, self.crc_length):
                    return u_hat, p["pm"]

        best = paths[0]
        u_hat = best["B"][:, self.n].astype(np.int8)
        return u_hat, best["pm"]
