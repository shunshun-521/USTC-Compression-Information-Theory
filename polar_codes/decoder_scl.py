"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from channel import permute_llr_for_decoder
from decoder_sc import (
    f_operation,
    g_operation,
    bit_reversed,
    active_llr_level,
    active_bit_level,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in info_bits:
        fb = ((reg >> (crc_length - 1)) ^ int(b)) & 1
        reg = ((reg << 1) & mask) ^ (poly if fb else 0)
        reg &= mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        fb = ((reg >> (crc_length - 1)) ^ int(b)) & 1
        reg = ((reg << 1) & mask) ^ (poly if fb else 0)
        reg &= mask
    return reg == 0


class SCLDecoder:
    """SCL 译码器（PSC 结构，路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        L[:, 0] = llr_ch
        return {"pm": 0.0, "u_hat": np.zeros(self.N, dtype=np.int8), "L": L, "B": B}

    def _clone(self, path):
        return {
            "pm": path["pm"],
            "u_hat": path["u_hat"].copy(),
            "L": path["L"].copy(),
            "B": path["B"].copy(),
        }

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
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
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    def _branch_pm(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return pm if u == u_hard else pm + abs(llr)

    def decode(self, llr_ch):
        llr_ch = permute_llr_for_decoder(llr_ch, self.N)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            expanded = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path["L"][l, self.n]
                if self.frozen_bits[l]:
                    child = self._clone(path)
                    child["pm"] = self._branch_pm(path["pm"], llr, 0)
                    child["B"][l, self.n] = 0
                    child["u_hat"][l] = 0
                    self._update_bits(child, l)
                    expanded.append(child)
                else:
                    for u in (0, 1):
                        child = self._clone(path)
                        child["pm"] = self._branch_pm(path["pm"], llr, u)
                        child["B"][l, self.n] = u
                        child["u_hat"][l] = u
                        self._update_bits(child, l)
                        expanded.append(child)
            expanded.sort(key=lambda p: p["pm"])
            paths = expanded[: self.L]

        crc_ok = []
        for p in paths:
            if self.crc_length > 0:
                if crc_check(p["u_hat"][self.info_indices], self.crc_length):
                    crc_ok.append(p)
        best = min(crc_ok if crc_ok else paths, key=lambda p: p["pm"])
        return best["u_hat"].astype(int), best["pm"]
