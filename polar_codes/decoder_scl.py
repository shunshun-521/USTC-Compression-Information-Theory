"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed,
    _update_path_bits,
    _update_path_llrs,
    sc_decode,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_run(bits, poly, crc_length):
    """按位反射 LFSR（CRC-8: poly=0x07, CRC-16: poly=0x8005）"""
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b)
        if reg & 1:
            reg = (reg >> 1) ^ poly
        else:
            reg >>= 1
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    rem = _crc_run(padded, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_run(bits, poly, crc_length) == 0


def _pm_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    return pm + (abs(llr) if u != u_hard else 0.0)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_perm = llr_ch[rev]
        n, N = self.n, self.N

        paths = [
            {
                "pm": 0.0,
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
                "u_hat": np.zeros(N, dtype=np.int8),
            }
        ]
        paths[0]["L"][:, 0] = llr_perm

        for i in range(N):
            l = _bit_reversed(i, n)
            expanded = []
            for path in paths:
                L_tmp = path["L"].copy()
                B_tmp = path["B"].copy()
                llr_leaf = _update_path_llrs(L_tmp, B_tmp, l, n, N)
                if l in self.frozen_set:
                    expanded.append((_pm_update(path["pm"], llr_leaf, 0), path, 0))
                else:
                    for u in (0, 1):
                        expanded.append((_pm_update(path["pm"], llr_leaf, u), path, u))

            expanded.sort(key=lambda x: x[0])
            expanded = expanded[: self.list_size]

            new_paths = []
            for pm, parent, u_bit in expanded:
                child = {
                    "pm": pm,
                    "L": parent["L"].copy(),
                    "B": parent["B"].copy(),
                    "u_hat": parent["u_hat"].copy(),
                }
                child["B"][l, n] = u_bit
                child["u_hat"][l] = u_bit
                _update_path_bits(child["B"], l, n, N)
                new_paths.append(child)
            paths = new_paths

        best = None
        if self.crc_length > 0:
            for path in paths:
                info_bits = path["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best is None or path["pm"] < best["pm"]:
                        best = path
        if best is None:
            best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), float(best["pm"])
