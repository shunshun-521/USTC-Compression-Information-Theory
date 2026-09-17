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
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    返回 True/False。
    """
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits[-crc_length:])


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def _pm_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def _copy_path(self, path):
        return {
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "pm": path["pm"],
            "u_hat": path["u_hat"].copy(),
        }

    def _init_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_ch
        return {"L": L, "B": B, "pm": 0.0, "u_hat": np.zeros(self.N, dtype=int)}

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = [self._init_path(llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n)
                llr = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    new_path = self._copy_path(path)
                    new_path["pm"] += self._pm_penalty(llr, 0)
                    new_path["B"][l, self.n] = 0
                    new_path["u_hat"][l] = 0
                    _update_bits(new_path["B"], l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path["pm"] += self._pm_penalty(llr, u_bit)
                        new_path["B"][l, self.n] = u_bit
                        new_path["u_hat"][l] = u_bit
                        _update_bits(new_path["B"], l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"], best["pm"]
