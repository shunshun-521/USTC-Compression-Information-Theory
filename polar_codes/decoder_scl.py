"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _SCDState, hard_decision
from encoder import bit_reversed_index

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return _CRC8_POLY, 8
    if crc_length == 16:
        return _CRC16_POLY, 16
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly, degree = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (degree - 1)
        for _ in range(8):
            if reg & (1 << (degree - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << degree) - 1)
            else:
                reg = (reg << 1) & ((1 << degree) - 1)
    crc_bits = np.array(
        [(reg >> (degree - 1 - i)) & 1 for i in range(degree)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly, degree = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (degree - 1)
        for _ in range(8):
            if reg & (1 << (degree - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << degree) - 1)
            else:
                reg = (reg << 1) & ((1 << degree) - 1)
    return reg == 0


class _Path:
    __slots__ = ("pm", "state", "u_hat")

    def __init__(self, N, n, llr_ch, frozen_set):
        self.pm = 0.0
        self.state = _SCDState(N, n, llr_ch, frozen_set)
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _penalty(self, llr, u):
        return 0.0 if u == hard_decision(llr) else abs(llr)

    def _clone(self, path):
        child = _Path(self.N, self.n, path.state.L[:, 0], self.frozen_set)
        child.pm = path.pm
        child.state.L = path.state.L.copy()
        child.state.B = path.state.B.copy()
        child.u_hat = path.u_hat.copy()
        return child

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch, self.frozen_set)]

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            new_paths = []
            for path in paths:
                path.state.update_llrs(l)
                llr = path.state.L[l, self.n]
                candidates = [0] if l in self.frozen_set else [0, 1]
                for u_bit in candidates:
                    child = self._clone(path)
                    child.pm += self._penalty(llr, u_bit)
                    child.state.B[l, self.n] = u_bit
                    child.u_hat[l] = u_bit
                    child.state.update_bits(l)
                    new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
