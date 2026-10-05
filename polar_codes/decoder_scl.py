"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import (
    bit_reversed,
    sc_decode,
    sc_new_state,
    sc_update_bits,
    sc_update_llrs,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _path_metric(llr_bit, u):
    if (u == 0 and llr_bit >= 0) or (u == 1 and llr_bit < 0):
        return 0.0
    return abs(llr_bit)


class SCLDecoder:
    """SCL 译码器（与 SC 共享 L/B 更新逻辑）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        N, n = self.N, self.n
        L0, B0, _ = sc_new_state(N, llr_ch)
        paths = [{"pm": 0.0, "L": L0, "B": B0}]

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []
            for path in paths:
                sc_update_llrs(path["L"], path["B"], l, n, N)
                llr_bit = path["L"][l, n]

                if self.frozen_bits[l]:
                    child = copy.deepcopy(path)
                    child["pm"] += _path_metric(llr_bit, 0)
                    child["B"][l, n] = 0
                    sc_update_bits(child["B"], l, n, N)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = copy.deepcopy(path)
                        child["pm"] += _path_metric(llr_bit, u)
                        child["B"][l, n] = u
                        sc_update_bits(child["B"], l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_hat = p["B"][:, n].astype(int)
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["B"][:, n].astype(int), best["pm"]
