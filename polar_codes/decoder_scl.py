"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    bit_reversed,
    active_llr_level,
    active_bit_level,
    _update_llrs,
    _update_bits,
    _map_frozen_to_decode_indices,
    f_operation,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
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
                reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ poly
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits.astype(int), crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(expected, bits[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "pm", "u_raw")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_raw = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = _map_frozen_to_decode_indices(frozen_bits, self.n)
        self.br = bit_reversal_permutation(N)

    def _pm_update(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def _advance_bit(self, paths, l, llr_val, u_bit):
        for p in paths:
            p.u_raw[l] = u_bit
            p.B[l, self.n] = u_bit
            p.pm = self._pm_update(p.pm, llr_val, u_bit)
            _update_bits(p.B, l, self.n)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [_Path(N, n) for _ in range(1)]
        paths[0].L[:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            active = []
            for p in paths:
                _update_llrs(p.L, p.B, l, n)
                llr_l = p.L[l, n]
                if l in self.frozen_set:
                    self._advance_bit([p], l, llr_l, 0)
                    active.append(p)
                else:
                    p0 = _Path(N, n)
                    p0.L = np.array(p.L, copy=True)
                    p0.B = np.array(p.B, copy=True)
                    p0.pm = p.pm
                    p0.u_raw = np.array(p.u_raw, copy=True)
                    self._advance_bit([p], l, llr_l, 0)
                    self._advance_bit([p0], l, llr_l, 1)
                    active.extend([p, p0])

            active.sort(key=lambda x: x.pm)
            paths = active[: self.list_size]

        best = min(paths, key=lambda x: x.pm)
        u_hat = best.u_raw[self.br]

        if self.crc_length > 0:
            info_idx = np.where(self.frozen_bits == 0)[0]
            info_idx = np.sort(info_idx)
            candidates = sorted(paths, key=lambda x: x.pm)
            for p in candidates:
                u_try = p.u_raw[self.br]
                payload = u_try[info_idx]
                if crc_check(payload, self.crc_length):
                    return u_try, p.pm
        return u_hat, best.pm
