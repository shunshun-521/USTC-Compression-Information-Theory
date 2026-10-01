"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Permuted SC 顺序
"""
import copy
import numpy as np

from decoder_sc import (
    _PermutedSCD,
    active_bit_level,
    active_llr_level,
    bit_reversed,
    hard_decision,
    lower_llr,
    upper_llr,
)
from encoder import bit_reversed as br_index


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= (int(b) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（路径复制版 Permuted SC）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_penalty(self, llr, bit):
        hard = hard_decision(llr)
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"pm": 0.0, "dec": _PermutedSCD(self.N, self.frozen_set)}]
        for p in paths:
            p["dec"].L[:, 0] = llr_ch

        order = [bit_reversed(i, self.n) for i in range(self.N)]

        for l in order:
            new_paths = []
            for path in paths:
                dec = path["dec"]
                dec.update_llrs(l)
                llr_bit = dec.L[l, dec.n]

                if l in self.frozen_set:
                    bit = 0
                    pm = path["pm"] + self._path_penalty(llr_bit, bit)
                    child = {"pm": pm, "dec": copy.deepcopy(dec)}
                    child["dec"].B[l, child["dec"].n] = 0
                    child["dec"].update_bits(l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + self._path_penalty(llr_bit, bit)
                        child = {"pm": pm, "dec": copy.deepcopy(dec)}
                        child["dec"].B[l, child["dec"].n] = bit
                        child["dec"].update_bits(l)
                        new_paths.append(child)

            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[: self.list_size]

        best = min(paths, key=lambda x: x["pm"])
        if self.crc_length > 0:
            valid = []
            for p in paths:
                u = p["dec"].B[:, self.n].astype(np.int8)
                if crc_check(u[self.info_indices], self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda x: x["pm"])

        u_hat = best["dec"].B[:, self.n].astype(np.int8)
        return u_hat, best["pm"]
