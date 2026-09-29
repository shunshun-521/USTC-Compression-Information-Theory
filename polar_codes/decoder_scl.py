"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    bit_reversed,
    upper_llr,
    lower_llr,
    _update_llr,
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
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, self.n] = llr_ch
        return {"pm": 0.0, "L": L, "B": B}

    @staticmethod
    def _pm_add(pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        return pm + (0.0 if bit == hard else abs(llr))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                _update_llr(path["L"], path["B"], l, self.n)
                llr = path["L"][l, 0]
                if l in self.frozen_set:
                    bit = 0
                    p2 = {
                        "pm": self._pm_add(path["pm"], llr, bit),
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                    }
                    p2["B"][l, 0] = bit
                    _update_bits(p2["B"], l, self.n)
                    new_paths.append(p2)
                else:
                    for bit in (0, 1):
                        p2 = {
                            "pm": self._pm_add(path["pm"], llr, bit),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        p2["B"][l, 0] = bit
                        _update_bits(p2["B"], l, self.n)
                        new_paths.append(p2)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info = p["B"][:, 0].astype(int)[self.info_indices]
                if crc_check(info, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["B"][:, 0].astype(int), best["pm"]
