"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _sc_decode_codeword
from encoder import polar_encode
from polar_ops import f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(info_bits, poly, crc_length):
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits[:-crc_length], poly, crc_length)
    expected = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.array_equal(bits[-crc_length:], expected)


def _pm_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


def _scl_paths(llr, frozen_bits, list_size):
    """在码字树上做 SCL，返回 (pm, codeword s) 列表"""

    def expand(paths, lam, frozen_leaf):
        size = len(lam)
        if size == 1:
            out = []
            for pm, _ in paths:
                if frozen_leaf[0]:
                    out.append((pm + _pm_penalty(lam[0], 0), np.array([0], dtype=int)))
                else:
                    for b in (0, 1):
                        out.append((pm + _pm_penalty(lam[0], b), np.array([b], dtype=int)))
            out.sort(key=lambda x: x[0])
            return out[:list_size]

        half = size // 2
        lam_left = f_operation(lam[:half], lam[half:])
        left_paths = expand(paths, lam_left, frozen_leaf[:half])

        merged = []
        for pm_l, s_left in left_paths:
            lam_right = g_operation(lam[:half], lam[half:], s_left)
            right_paths = expand([(pm_l, None)], lam_right, frozen_leaf[half:])
            for pm_r, s_right in right_paths:
                s = np.zeros(size, dtype=int)
                s[:half] = s_left ^ s_right
                s[half:] = s_right
                merged.append((pm_r, s))
        merged.sort(key=lambda x: x[0])
        return merged[:list_size]

    return expand([(0.0, None)], llr, frozen_bits)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            s = _sc_decode_codeword(llr_ch, self.frozen_bits)
            u = polar_encode(s)
            return u, 0.0

        paths = _scl_paths(llr_ch, self.frozen_bits, self.list_size)
        candidates = []
        for pm, s in paths:
            u = polar_encode(s)
            candidates.append((pm, u))

        if self.crc_length > 0:
            valid = [(pm, u) for pm, u in candidates if crc_check(u, self.crc_length)]
            if valid:
                pm, u = min(valid, key=lambda x: x[0])
                return u, pm

        pm, u = candidates[0]
        return u, pm
