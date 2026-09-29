"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, crc_bits]).astype(int)


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码（基于 Permuted SCD 树，路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.Lsize = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _pm_update(self, pm, llr, u):
        penalty = 0.0 if (llr >= 0 and u == 0) or (llr < 0 and u == 1) else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        inv_br = np.argsort(bit_reversal_permutation(self.N))
        llr_v = np.asarray(llr_ch, dtype=np.float64)[inv_br]

        paths = [
            {
                "L": np.full((self.N, self.n + 1), np.nan),
                "B": np.zeros((self.N, self.n + 1), dtype=int),
                "pm": 0.0,
                "u": np.zeros(self.N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_v

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []
            for p in paths:
                self._update_llrs_path(p, l)
                llr = p["L"][l, self.n]
                if l in self.frozen_set:
                    u_bit = 0
                    p["B"][l, self.n] = 0
                    p["pm"] = self._pm_update(p["pm"], llr, u_bit)
                    p["u"][l] = u_bit
                    self._update_bits_path(p, l)
                    new_paths.append(p)
                else:
                    for u_bit in (0, 1):
                        cp = {
                            "L": p["L"].copy(),
                            "B": p["B"].copy(),
                            "pm": self._pm_update(p["pm"], llr, u_bit),
                            "u": p["u"].copy(),
                        }
                        cp["B"][l, self.n] = u_bit
                        cp["u"][l] = u_bit
                        self._update_bits_path(cp, l)
                        new_paths.append(cp)
            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[: self.Lsize]

        best = paths[0]
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"][self.info_indices], self.crc_length)]
            if valid:
                best = min(valid, key=lambda x: x["pm"])
        return best["u"].astype(int), float(best["pm"])

    def _update_llrs_path(self, p, l):
        L, B = p["L"], p["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            bs = 1 << (s + 1)
            t = bs >> 1
            for j in range(l, self.N, bs):
                if j % bs < t:
                    L[j, s + 1] = f_operation(L[j, s], L[j + t, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - t, s], L[j, s], int(B[j - t, s + 1])
                    )

    def _update_bits_path(self, p, l):
        B = p["B"]
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            bs = 1 << s
            t = bs >> 1
            for j in range(l, -1, -bs):
                if j % bs >= t:
                    B[j - t, s - 1] = int(B[j, s]) ^ int(B[j - t, s])
                    B[j, s - 1] = B[j, s]
