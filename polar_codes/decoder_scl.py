"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _update_bits,
    _update_llrs,
    active_bit_level,
    active_llr_level,
    bit_reversed_index,
)


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    poly = _crc_polynomial(crc_length)
    info_bits = np.asarray(info_bits, dtype=np.int8)
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
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


def _llr_to_bit(llr):
    return 0 if llr >= 0 else 1


def _pm_penalty(llr, u):
    hard = _llr_to_bit(llr)
    return 0.0 if u == hard else abs(llr)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)

            cur_llrs = [path.L[l, self.n] for path in paths]
            new_paths = []

            if self.frozen_bits[l]:
                for path, llr in zip(paths, cur_llrs):
                    path.pm += _pm_penalty(llr, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    _update_bits(path.B, l, self.n, self.N)
                    new_paths.append(path)
            else:
                for path, llr in zip(paths, cur_llrs):
                    for u in (0, 1):
                        child = _Path(self.N, self.n, llr_ch)
                        child.pm = path.pm + _pm_penalty(llr, u)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[l] = u
                        child.B[l, self.n] = u
                        _update_bits(child.B, l, self.n, self.N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        crc_valid = []
        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[~self.frozen_bits]
                crc_valid.append(crc_check(info_bits, self.crc_length))
            else:
                crc_valid.append(True)

        candidates = [p for p, ok in zip(paths, crc_valid) if ok] or paths
        best = min(candidates, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
