"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
    sc_decode,
)
from encoder import bit_reversal_permutation


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_division(bits, poly, crc_length):
    reg = np.zeros(crc_length, dtype=np.int8)
    for bit in bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            poly_bits = np.array(
                [(poly >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
                dtype=np.int8,
            )
            reg ^= poly_bits
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_division(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_division(bits, poly, crc_length)
    return np.all(remainder == 0)


class SCLDecoder:
    """SCL 译码器（PSCD 结构 + 路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        L[:, 0] = llr_ch[self.rev]
        return {"L": L, "B": B, "pm": 0.0}

    def _branch_penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n)
                llr_val = path["L"][l, self.n]

                if l in self.frozen_set:
                    pm = path["pm"] + self._branch_penalty(llr_val, 0)
                    new_path = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": pm,
                    }
                    new_path["B"][l, self.n] = 0
                    _update_bits(new_path["B"], l, self.n)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + self._branch_penalty(llr_val, bit)
                        new_path = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": pm,
                        }
                        new_path["B"][l, self.n] = bit
                        _update_bits(new_path["B"], l, self.n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path["B"][:, self.n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid, key=lambda p: p["pm"]) if valid else paths[0]
        else:
            best = paths[0]

        return best["B"][:, self.n].astype(int), best["pm"]
