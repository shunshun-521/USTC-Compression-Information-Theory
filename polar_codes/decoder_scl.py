"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import _update_bits, _update_llrs
from encoder import bit_reversed_index


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return _CRC8_POLY
    if crc_length == 16:
        return _CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def _crc_update(reg, bit, crc_length, poly):
    reg ^= int(bit) << (crc_length - 1)
    if reg & (1 << (crc_length - 1)):
        reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
    else:
        reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg = _crc_update(reg, bit, crc_length, poly)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg = _crc_update(reg, bit, crc_length, poly)
    return reg == 0


def _pm_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [
            {
                "pm": 0.0,
                "u": np.zeros(self.N, dtype=np.int8),
                "L": np.column_stack([llr_ch, np.full((self.N, self.n), np.nan)]),
                "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
            }
        ]

        for phi_nat in range(self.N):
            l = bit_reversed_index(phi_nat, self.n)
            candidates = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n)
                llr_val = path["L"][l, self.n]
                if self.frozen_bits[l]:
                    new_path = copy.deepcopy(path)
                    new_path["pm"] += _pm_penalty(llr_val, 0)
                    new_path["u"][l] = 0
                    new_path["B"][l, self.n] = 0
                    _update_bits(new_path["B"], l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = copy.deepcopy(path)
                        new_path["pm"] += _pm_penalty(llr_val, u)
                        new_path["u"][l] = u
                        new_path["B"][l, self.n] = u
                        _update_bits(new_path["B"], l, self.n, self.N)
                        candidates.append(new_path)
            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u"][self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u"].astype(int), best["pm"]
