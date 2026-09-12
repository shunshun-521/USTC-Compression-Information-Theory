"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import _update_bits, _update_llrs
from encoder import bit_reversed


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
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（置换 SC 框架）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    @staticmethod
    def _pm_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, Lsize = self.N, self.n, self.list_size

        paths = [{
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.full((N, n + 1), np.nan),
            "u_hat": np.zeros(N, dtype=int),
            "pm": 0.0,
        }]
        paths[0]["L"][:, 0] = llr_ch.copy()

        for phi in range(N):
            l = bit_reversed(phi, n)
            new_paths = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, n, N)
                cur_llr = path["L"][l, n]

                if l in self.frozen_set:
                    opts = [(0, path["pm"] + self._pm_penalty(cur_llr, 0))]
                else:
                    opts = [
                        (0, path["pm"] + self._pm_penalty(cur_llr, 0)),
                        (1, path["pm"] + self._pm_penalty(cur_llr, 1)),
                    ]

                for u, pm in opts:
                    child = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u_hat": path["u_hat"].copy(),
                        "pm": pm,
                    }
                    child["B"][l, n] = u
                    child["u_hat"][l] = u
                    _update_bits(child["B"], l, n, N)
                    new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:Lsize]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"], best["pm"]
