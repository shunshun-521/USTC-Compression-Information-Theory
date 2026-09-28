"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Permuted SCD 树
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    sc_decode,
    _bit_reversed,
    _update_llrs,
    _update_bits,
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
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    """SCL 译码器（每条路径维护独立 L/B 矩阵）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path_state(self, llr_ch):
        br = bit_reversal_permutation(self.N)
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_ch[br]
        return {"pm": 0.0, "L": L, "B": B, "u": np.full(self.N, -1, dtype=int)}

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [self._new_path_state(llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []
            for p in paths:
                L = p["L"]
                B = p["B"]
                _update_llrs(L, B, l, self.n, self.N)
                llr = L[l, self.n]
                if l in self.frozen_set:
                    u_bit = 0
                    penalty = abs(llr) if llr < 0 else 0.0
                    candidates.append((p["pm"] + penalty, p, u_bit))
                else:
                    for u_bit in (0, 1):
                        ok = (u_bit == 0 and llr >= 0) or (u_bit == 1 and llr < 0)
                        penalty = 0.0 if ok else abs(llr)
                        candidates.append((p["pm"] + penalty, p, u_bit))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.L]

            new_paths = []
            for pm, parent, u_bit in candidates:
                L = parent["L"].copy()
                B = parent["B"].copy()
                B[l, self.n] = u_bit
                _update_bits(B, l, self.n, self.N)
                u = parent["u"].copy()
                u[l] = u_bit
                new_paths.append({"pm": pm, "L": L, "B": B, "u": u})
            paths = new_paths

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u"][self.info_indices], self.crc_length)
            ]
            pool = valid if valid else paths
        else:
            pool = paths
        best = min(pool, key=lambda p: p["pm"])
        return best["u"].astype(int), best["pm"]
