"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Permuted SC 结构 + 路径列表）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits.astype(bool))[0])
        self.info_indices = np.where(~self.frozen_bits.astype(bool))[0]

    def _pm_penalty(self, llr_val, u_bit):
        u_hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == u_hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        rev = bit_reversal_permutation(N)
        llr_ch = llr_ch[rev]

        paths = [
            {
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
                "pm": 0.0,
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []
            for path in paths:
                L = path["L"]
                B = path["B"]
                _update_llrs(L, B, l, n)
                llr0 = L[l, n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(llr0, 0)
                    child = copy.deepcopy(path)
                    child["pm"] += pen
                    child["B"][l, n] = 0
                    _update_bits(child["B"], l, n)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        pen = self._pm_penalty(llr0, u_bit)
                        child = copy.deepcopy(path)
                        child["pm"] += pen
                        child["B"][l, n] = u_bit
                        _update_bits(child["B"], l, n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = self._select_path(paths)
        return best["B"][:, n].astype(int), best["pm"]

    def _select_path(self, paths):
        if self.crc_length > 0:
            passed = []
            for p in paths:
                info_bits = p["B"][:, self.n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    passed.append(p)
            if passed:
                return min(passed, key=lambda x: x["pm"])
        return min(paths, key=lambda x: x["pm"])
