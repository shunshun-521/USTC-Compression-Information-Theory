"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助 CA-SCL
"""
import math
import numpy as np
from decoder_sc import sc_decode
from decoder_utils_local import bit_reversed, hard_decision, upper_llr, lower_llr
from decoder_utils_local import active_llr_level, active_bit_level


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    for b in np.asarray(info_bits, dtype=int):
        reg ^= (b << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([np.asarray(info_bits, dtype=int), crc_bits])


def crc_check(bits, crc_length=8):
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, np.asarray(bits, dtype=int))


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.active = True


class SCLDecoder:
    """SCL 译码器（路径复制实现，稳定优先）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        fb = np.asarray(frozen_bits)
        if fb.dtype == bool:
            self.frozen = set(np.where(fb)[0])
        else:
            self.frozen = set(np.where(fb.astype(int) == 1)[0])

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            bs = 2 ** (s + 1)
            br = bs // 2
            for j in range(l, self.N, bs):
                if j % bs < br:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + br, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j - br, s],
                        path.L[j, s],
                        int(path.B[j - br, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            bs = 2 ** s
            br = bs // 2
            for j in range(l, -1, -bs):
                if j % bs >= br:
                    path.B[j - br, s - 1] = int(path.B[j, s]) ^ int(path.B[j - br, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _llr_bit(self, path, i):
        l = bit_reversed(i, self.n)
        self._update_llrs(path, l)
        return path.L[l, self.n]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]
        for i in range(self.N):
            new_paths = []
            for p in paths:
                llr_i = self._llr_bit(p, i)
                if i in self.frozen:
                    cand = [(0, p.pm + (abs(llr_i) if llr_i < 0 else 0.0))]
                else:
                    pm0 = p.pm + (0.0 if llr_i >= 0 else abs(llr_i))
                    pm1 = p.pm + (0.0 if llr_i < 0 else abs(llr_i))
                    cand = [(0, pm0), (1, pm1)]
                for u_bit, pm in cand:
                    cp = _Path(self.N, self.n, llr_ch)
                    cp.L = p.L.copy()
                    cp.B = p.B.copy()
                    cp.u_hat = p.u_hat.copy()
                    cp.pm = pm
                    l = bit_reversed(i, self.n)
                    cp.u_hat[i] = u_bit
                    cp.B[l, self.n] = u_bit
                    self._update_bits(cp, l)
                    new_paths.append(cp)
            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            pick = min(valid, key=lambda x: x.pm) if valid else min(paths, key=lambda x: x.pm)
        else:
            pick = min(paths, key=lambda x: x.pm)
        return pick.u_hat.astype(int), pick.pm


def scl_equivalent_sc(llr, frozen_bits):
    """L=1 且不用 CRC 时应接近 SC"""
    dec = SCLDecoder(len(llr), frozen_bits, list_size=1, crc_length=0)
    u, _ = dec.decode(llr)
    return u
