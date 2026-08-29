"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
)


_CRC8_POLY = np.array([1, 1, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8)
_CRC16_POLY = np.array(
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1], dtype=np.int8
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return _CRC8_POLY
    if crc_length == 16:
        return _CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    r = crc_length
    reg = np.zeros(r, dtype=np.int8)
    for bit in info_bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg ^= poly[1:]
    return np.concatenate([info_bits, reg])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    r = crc_length
    reg = np.zeros(r, dtype=np.int8)
    for bit in bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg ^= poly[1:]
    return np.all(reg == 0)


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.pm = 0.0

    def copy(self):
        p = _Path(self.L.shape[0], self.L.shape[1] - 1)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        return p


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self._decode_order = [bit_reversed_index(i, self.n) for i in range(N)]
        self._info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            bs = 2 ** (s + 1)
            br = bs // 2
            for j in range(l, self.N, bs):
                if j % bs < br:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + br, s])
                else:
                    tb = int(path.B[j - br, s + 1])
                    path.L[j, s + 1] = _lower_llr(path.L[j, s], path.L[j - br, s], tb)

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            bs = 2 ** s
            br = bs // 2
            for j in range(l, -1, -bs):
                if j % bs >= br:
                    path.B[j - br, s - 1] = int(path.B[j, s]) ^ int(path.B[j - br, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        root = _Path(self.N, self.n)
        root.L[:, 0] = llr_ch
        paths = [root]

        for l in self._decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path.L[l, self.n]
                if self.frozen_bits[l]:
                    child = path.copy()
                    child.pm += self._penalty(llr_bit, 0)
                    child.B[l, self.n] = 0
                    self._update_bits(child, l)
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm += self._penalty(llr_bit, bit)
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        candidates.append(child)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = p.B[:, self.n].astype(np.int8)
                if crc_check(bits[self._info_indices], self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        u_hat = best.B[:, self.n].astype(np.int8)
        return u_hat, best.pm

