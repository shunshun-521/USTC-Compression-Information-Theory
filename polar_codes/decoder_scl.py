"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _SCD
from encoder import bit_reversed


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
        reg <<= 1
        reg |= int(b)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class SCLDecoder:
    """SCL 译码器（路径级复制 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_penalty(self, llr, u):
        if (u == 0 and llr >= 0) or (u == 1 and llr < 0):
            return 0.0
        return abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
                "u": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        scd = _SCD(N, np.array([i in self.frozen for i in range(N)], dtype=bool))

        for dec_i in range(N):
            l = bit_reversed(dec_i, n)
            candidates = []

            for p_idx, path in enumerate(paths):
                scd.L = path["L"]
                scd.B = path["B"]
                scd.update_llrs(l)
                llr = scd.L[l, n]

                if l in self.frozen:
                    pen = self._path_metric_penalty(llr, 0)
                    candidates.append((path["pm"] + pen, p_idx, 0))
                else:
                    for u in (0, 1):
                        pen = self._path_metric_penalty(llr, u)
                        candidates.append((path["pm"] + pen, p_idx, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, p_idx, u in candidates:
                base = paths[p_idx]
                np_path = {
                    "pm": pm,
                    "L": base["L"].copy(),
                    "B": base["B"].copy(),
                    "u": base["u"].copy(),
                }
                np_path["u"][l] = u
                scd.L = np_path["L"]
                scd.B = np_path["B"]
                scd.B[l, n] = u
                scd.update_bits(l)
                new_paths.append(np_path)
            paths = new_paths

        best = min(paths, key=lambda p: p["pm"])
        if self.crc_length > 0:
            crc_paths = [p for p in paths if crc_check(p["u"], self.crc_length)]
            if crc_paths:
                best = min(crc_paths, key=lambda p: p["pm"])

        return best["u"], best["pm"]
