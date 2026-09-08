"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    upper_llr,
    lower_llr,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _bits_to_int(bits):
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def _int_to_bits(val, length):
    return np.array([(val >> (length - 1 - i)) & 1 for i in range(length)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    poly_full = (1 << crc_length) | poly
    msg = _bits_to_int(info_bits) << crc_length
    nbits = len(info_bits) + crc_length
    for i in range(len(info_bits) - 1, -1, -1):
        if (msg >> (nbits - 1 - i)) & 1:
            msg ^= poly_full << i
    crc_bits = _int_to_bits(msg & ((1 << crc_length) - 1), crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    poly_full = (1 << crc_length) | poly
    msg = _bits_to_int(bits)
    nbits = len(bits)
    for i in range(nbits - crc_length - 1, -1, -1):
        if (msg >> (nbits - 1 - i)) & 1:
            msg ^= poly_full << i
    return (msg & ((1 << crc_length) - 1)) == 0


class Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_path(self, src):
        dst = Path(self.N, self.n)
        dst.pm = src.pm
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.u_hat = src.u_hat.copy()
        return dst

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 1 << (s + 1)
            half = block >> 1
            for j in range(l, self.N, block):
                if j % block < half:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + half, s])
                else:
                    top_bit = (
                        0 if np.isnan(path.B[j - half, s + 1]) else int(path.B[j - half, s + 1])
                    )
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s], path.L[j - half, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 1 << s
            half = block >> 1
            for j in range(l, -1, -block):
                if j % block >= half:
                    prev = 0 if np.isnan(path.B[j - half, s]) else int(path.B[j - half, s])
                    path.B[j - half, s - 1] = int(path.B[j, s]) ^ prev
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return abs(llr) if bit != hard else 0.0

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch.copy()

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if np.isnan(llr):
                    llr = 0.0

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p = self._copy_path(path)
                        p.pm += self._pm_penalty(llr, bit)
                        p.u_hat[l] = bit
                        p.B[l, self.n] = bit
                        self._update_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
