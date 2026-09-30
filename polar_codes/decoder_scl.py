"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    bit_reversed,
    upper_llr,
    lower_llr,
    active_llr_level,
    active_bit_level,
    _update_llrs,
    _update_bits,
)


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位（CRC-8: 0x07, CRC-16: 0x8005）"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


class SCLDecoder:
    """SCL 译码器（Permuted SC + 路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        if crc_length > 0:
            self.info_positions = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_ch
        return {"pm": 0.0, "L": L, "B": B}

    def _pm_add(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return pm + (0.0 if u == u_hard else abs(llr))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n)
                llr_bit = path["L"][l, self.n]
                if self.frozen_bits[l]:
                    pm = self._pm_add(path["pm"], llr_bit, 0)
                    child = {
                        "pm": pm,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                    }
                    child["B"][l, self.n] = 0
                    _update_bits(child["B"], l, self.n, self.N)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        pm = self._pm_add(path["pm"], llr_bit, u)
                        child = {
                            "pm": pm,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        child["B"][l, self.n] = u
                        _update_bits(child["B"], l, self.n, self.N)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        u_hat = paths[0]["B"][:, self.n].astype(int)
        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = p["B"][:, self.n].astype(int)[self.info_positions]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p["pm"])
            else:
                best = paths[0]
        else:
            best = paths[0]
        return best["B"][:, self.n].astype(int), best["pm"]
