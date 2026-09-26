"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from encoder import bit_reversed
from polarcodes_mini.decoder_utils import (
    active_bit_level,
    active_llr_level,
    lower_llr,
    upper_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = 0
    mask = (1 << crc_length) - 1
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    poly = _crc_poly(crc_length)
    bits = np.asarray(bits, dtype=np.int8)
    reg = 0
    mask = (1 << crc_length) - 1
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg == 0


class SCLDecoder:
    """SCL 译码器（路径复制，LLR/比特更新与 SCD 相同）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
        self.info_indices = np.where(np.asarray(frozen_bits, dtype=int) == 0)[0]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        L[:, 0] = llr_ch
        return {
            "pm": 0.0,
            "L": L,
            "B": np.full((self.N, self.n + 1), np.nan),
            "u": np.zeros(self.N, dtype=int),
        }

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        B = path["B"]
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            expanded = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path["L"][l, self.n]
                if l in self.frozen:
                    p = copy.deepcopy(path)
                    p["pm"] += self._penalty(llr, 0)
                    p["B"][l, self.n] = 0
                    p["u"][l] = 0
                    self._update_bits(p, l)
                    expanded.append(p)
                else:
                    for u in (0, 1):
                        p = copy.deepcopy(path)
                        p["pm"] += self._penalty(llr, u)
                        p["B"][l, self.n] = u
                        p["u"][l] = u
                        self._update_bits(p, l)
                        expanded.append(p)
            expanded.sort(key=lambda p: p["pm"])
            paths = expanded[: self.list_size]

        best_crc = None
        best = paths[0]
        for p in paths:
            if p["pm"] < best["pm"]:
                best = p
            if self.crc_length > 0:
                info = p["u"][self.info_indices]
                if crc_check(info, self.crc_length):
                    if best_crc is None or p["pm"] < best_crc["pm"]:
                        best_crc = p

        chosen = best_crc if best_crc is not None else best
        return chosen["u"].astype(int), chosen["pm"]
