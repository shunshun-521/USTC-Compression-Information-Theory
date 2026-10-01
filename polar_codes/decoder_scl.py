"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
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
    reg = 0
    for bit in info_bits:
        fb = (reg >> (crc_length - 1)) ^ int(bit)
        reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ (fb * poly)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(
        crc_encode(bits[:-crc_length], crc_length)[-crc_length:], bits[-crc_length:]
    )


def _sc_init_state(llr_ch, N, n):
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = llr_ch
    return L, B


def _sc_update_llrs(L, B, l, N, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                top_bit = int(B[j - branch_size, s + 1])
                L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)


def _sc_update_bits(B, l, N, n):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _penalty(llr, u):
    v = 0 if llr >= 0 else 1
    return 0.0 if u == v else abs(llr)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [
            {
                "L": _sc_init_state(llr_ch.copy(), N, n)[0],
                "B": _sc_init_state(llr_ch, N, n)[1],
                "pm": 0.0,
                "u": np.zeros(N, dtype=np.int8),
            }
        ]

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []
            for path in paths:
                L, B = path["L"], path["B"]
                _sc_update_llrs(L, B, l, N, n)
                llr_leaf = L[l, n]

                if l in self.frozen_set:
                    pm = path["pm"] + _penalty(llr_leaf, 0)
                    B[l, n] = 0
                    _sc_update_bits(B, l, N, n)
                    u = path["u"].copy()
                    u[l] = 0
                    new_paths.append({"L": L, "B": B, "pm": pm, "u": u})
                else:
                    for u_bit in (0, 1):
                        Lc = L.copy()
                        Bc = B.copy()
                        pm = path["pm"] + _penalty(llr_leaf, u_bit)
                        Bc[l, n] = u_bit
                        _sc_update_bits(Bc, l, N, n)
                        u = path["u"].copy()
                        u[l] = u_bit
                        new_paths.append({"L": Lc, "B": Bc, "pm": pm, "u": u})
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.L_size]

        best = paths[0]
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = [
                p for p in paths if crc_check(p["u"][info_idx], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p["pm"])

        return best["u"].astype(int), float(best["pm"])
