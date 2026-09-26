"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
    sc_decode,
)
from channel import align_llr_to_decoder
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for b in np.asarray(info_bits, dtype=int):
        reg ^= (b << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([np.asarray(info_bits, dtype=int), crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=int)

    def fork(self):
        q = _Path.__new__(_Path)
        q.pm = self.pm
        q.L = self.L.copy()
        q.B = self.B.copy()
        q.u_hat = self.u_hat.copy()
        return q


class SCLDecoder:
    """SCL 译码器（路径复制 L/B 阵列）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.brp = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = align_llr_to_decoder(np.asarray(llr_ch, dtype=np.float64))
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = self.brp[i]
            new_paths = []
            for path in paths:
                work = path.fork()
                _update_llrs(work.L, work.B, l, self.n)
                llr = work.L[l, self.n]

                if self.frozen_bits[l]:
                    p = work
                    p.pm += abs(llr) if llr < 0 else 0.0
                    p.u_hat[l] = 0
                    p.B[l, self.n] = 0
                    _update_bits(p.B, l, self.n, self.N)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = work.fork()
                        llr_p = p.L[l, self.n]
                        if (u == 0 and llr_p < 0) or (u == 1 and llr_p >= 0):
                            p.pm += abs(llr_p)
                        p.u_hat[l] = u
                        p.B[l, self.n] = u
                        _update_bits(p.B, l, self.n, self.N)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            if valid:
                paths = valid

        best = paths[0]
        return best.u_hat.copy(), best.pm


def scl_decode_equivalent_sc(llr_ch, frozen_bits):
    dec = SCLDecoder(len(llr_ch), frozen_bits, list_size=1, crc_length=0)
    u, _ = dec.decode(llr_ch)
    return np.array_equal(u, sc_decode(llr_ch, frozen_bits))
