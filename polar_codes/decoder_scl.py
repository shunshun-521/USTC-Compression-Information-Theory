"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
from __future__ import annotations

import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _update_bits,
    _update_llrs,
    _upper_llr,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)。"""
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits.astype(int), crc_bits])


def crc_check(bits, crc_length=8):
    """校验末尾 crc_length 位。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 0:
        return True
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def _pm_update(pm: float, llr: float, u: int) -> float:
    hard = 0 if llr >= 0 else 1
    if u == hard:
        return pm
    return pm + abs(llr)


class _Path:
    __slots__ = ("L", "B", "pm", "u")

    def __init__(self, N: int, n: int, llr_ch: np.ndarray):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u = np.zeros(N, dtype=int)

    def copy(self):
        p = _Path.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u = self.u.copy()
        return p


class SCLDecoder:
    """SCL 译码（路径复制；L 较小时开销可接受）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths: list[_Path] = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]
                if self.frozen_bits[l]:
                    path.pm = _pm_update(path.pm, llr, 0)
                    path.u[l] = 0
                    path.B[l, self.n] = 0
                    _update_bits(path.B, l, self.n)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm = _pm_update(child.pm, llr, bit)
                        child.u[l] = bit
                        child.B[l, self.n] = bit
                        _update_bits(child.B, l, self.n)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u, self.crc_length)]
            if valid:
                paths = valid
        best = min(paths, key=lambda p: p.pm)
        return best.u.copy(), best.pm
