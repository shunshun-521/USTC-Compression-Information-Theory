"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
)


CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    poly = CRC_POLYS[crc_length]
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([np.asarray(info_bits, dtype=int), crc_bits])


def crc_check(bits, crc_length=8):
    poly = CRC_POLYS[crc_length]
    return _crc_remainder(bits, poly, crc_length) == 0


class Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, n, N):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1))
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=int)
            if info_indices is not None
            else np.where(~self.frozen_bits)[0]
        )

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        N = self.N
        n = self.n

        br = bit_reversal_permutation(N)
        paths = [Path(n, N)]
        paths[0].L[:, 0] = llr_ch[br]

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n, N)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr, 0)
                    path.B[l, n] = 0
                    path.u_hat[l] = 0
                    _update_bits(path.B, l, n, N)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = Path(n, N)
                        child.pm = path.pm + self._pm_penalty(llr, bit)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.u_hat = path.u_hat.copy()
                        child.B[l, n] = bit
                        child.u_hat[l] = bit
                        _update_bits(child.B, l, n, N)
                        new_paths.append(child)

            if l not in self.frozen_set:
                new_paths.sort(key=lambda p: p.pm)
                new_paths = new_paths[: self.list_size]
            paths = new_paths

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
