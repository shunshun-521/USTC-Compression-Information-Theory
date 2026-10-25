"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    bit_reversed_index,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_bits_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_bits_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_bits_remainder(bits, poly, crc_length)
    return rem == 0


class _Path:
    __slots__ = ('pm', 'L', 'B', 'u_hat')

    def __init__(self, N, n, llr_work):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_work
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径级 Lazy Copy：分裂时复制 L/B）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.brp = bit_reversal_permutation(N)

    def _pm_update(self, pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_work = llr_ch[self.brp]
        paths = [_Path(self.N, self.n, llr_work)]

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr_bit = path.L[l, self.n]

                if l in self.frozen_set:
                    pm = self._pm_update(path.pm, llr_bit, 0)
                    child = _Path(self.N, self.n, llr_work)
                    child.pm = pm
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.u_hat = path.u_hat.copy()
                    child.B[l, self.n] = 0
                    child.u_hat[l] = 0
                    _update_bits(child.B, l, self.n)
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        pm = self._pm_update(path.pm, llr_bit, u)
                        child = _Path(self.N, self.n, llr_work)
                        child.pm = pm
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.u_hat = path.u_hat.copy()
                        child.B[l, self.n] = u
                        child.u_hat[l] = u
                        _update_bits(child.B, l, self.n)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            for path in paths:
                if self._crc_pass(path.u_hat):
                    return path.u_hat, path.pm
        return paths[0].u_hat, paths[0].pm

    def _crc_pass(self, u_nat):
        bits = u_nat[self.info_indices]
        return crc_check(bits, self.crc_length)
