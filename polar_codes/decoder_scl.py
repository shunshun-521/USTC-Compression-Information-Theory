"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
    bit_reversed,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（路径复制 L/B 矩阵）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_penalty(self, llr, u):
        u_from_llr = 0 if llr >= 0 else 1
        return 0.0 if u == u_from_llr else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        rev = bit_reversal_permutation(N)
        llr_ch = llr_ch[rev]

        paths = []
        L0 = np.zeros((N, n + 1), dtype=np.float64)
        L0[:, 0] = llr_ch
        B0 = np.zeros((N, n + 1), dtype=int)
        paths.append({"pm": 0.0, "L": L0, "B": B0})

        for phi in range(N):
            l = bit_reversed(phi, n)
            new_paths = []
            for path in paths:
                L = path["L"]
                B = path["B"]
                _update_llrs(L, B, l, n, N)
                llr = L[l, n]

                if self.frozen_bits[l]:
                    pen = self._pm_penalty(llr, 0)
                    B[l, n] = 0
                    Bc = B.copy()
                    _update_bits(Bc, l, n, N)
                    new_paths.append(
                        {"pm": path["pm"] + pen, "L": L.copy(), "B": Bc}
                    )
                else:
                    for u in (0, 1):
                        Lc = L.copy()
                        Bc = B.copy()
                        Bc[l, n] = u
                        pen = self._pm_penalty(llr, u)
                        _update_bits(Bc, l, n, N)
                        new_paths.append(
                            {"pm": path["pm"] + pen, "L": Lc, "B": Bc}
                        )

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            for path in paths:
                u_hat = path["B"][:, n].astype(int)
                info_part = u_hat[~self.frozen_bits]
                if crc_check(info_part, self.crc_length):
                    return u_hat.copy(), path["pm"]

        best = paths[0]
        u_hat = best["B"][:, n].astype(int)
        return u_hat.copy(), best["pm"]
