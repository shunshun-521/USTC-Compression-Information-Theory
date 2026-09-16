"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
)
from encoder import bit_reversal_permutation


CRC8_POLY = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
CRC16_POLY = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int)


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    r = crc_length
    reg = np.zeros(r, dtype=int)
    for bit in info_bits:
        fb = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if fb:
            reg ^= poly[1:]
    return np.concatenate([info_bits, reg])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


def _llr_to_bit(llr):
    return 0 if llr >= 0 else 1


def _pm_update(pm, llr, u):
    if u != _llr_to_bit(llr):
        pm += abs(llr)
    return pm


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, n, N):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = _Path(self.L.shape[1] - 1, self.L.shape[0])
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _advance_path(self, path, l):
        _update_llrs(path.L, path.B, l, self.n)

    def _propagate_bits(self, path, l):
        _update_bits(path.B, l, self.n)
        path.u_hat = path.B[:, self.n].astype(int)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        active = [_Path(self.n, self.N)]
        active[0].L[:, 0] = llr_ch[rev]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in active:
                self._advance_path(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = path.copy()
                    new_path.pm = _pm_update(new_path.pm, llr_val, 0)
                    new_path.B[l, self.n] = 0
                    self._propagate_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = path.copy()
                        new_path.pm = _pm_update(new_path.pm, llr_val, u)
                        new_path.B[l, self.n] = u
                        self._propagate_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            active = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in active:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                active = valid

        best = active[0]
        return best.u_hat.copy(), best.pm
