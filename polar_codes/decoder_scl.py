"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _update_bits,
    _update_llrs,
    sc_decode,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """LFSR CRC（MSB first），返回信息比特 + CRC。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) ^ int(bit)
        reg = ((reg << 1) & mask) ^ (poly if msb else 0)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        msb = (reg >> (crc_length - 1)) ^ int(bit)
        reg = ((reg << 1) & mask) ^ (poly if msb else 0)
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0


class SCLDecoder:
    """SCL 译码器（置换 SC + 路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    @staticmethod
    def _pm_penalty(llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr_dec = path.L[l, self.n]

                if l in self.frozen_set:
                    new_p = _Path(self.N, self.n, llr_ch)
                    new_p.L = path.L.copy()
                    new_p.B = path.B.copy()
                    new_p.pm = path.pm + self._pm_penalty(llr_dec, 0)
                    new_p.B[l, self.n] = 0
                    _update_bits(new_p.B, l, self.n)
                    candidates.append(new_p)
                else:
                    for u_bit in (0, 1):
                        new_p = _Path(self.N, self.n, llr_ch)
                        new_p.L = path.L.copy()
                        new_p.B = path.B.copy()
                        new_p.pm = path.pm + self._pm_penalty(llr_dec, u_bit)
                        new_p.B[l, self.n] = u_bit
                        _update_bits(new_p.B, l, self.n)
                        candidates.append(new_p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.B[:, self.n][self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)

        u_hat = best.B[:, self.n].astype(int)
        return u_hat, best.pm


def scl_decode_equivalent_sc(llr_ch, frozen_bits):
    """L=1 时应与 SC 一致。"""
    dec = SCLDecoder(len(llr_ch), frozen_bits, list_size=1)
    u, _ = dec.decode(llr_ch)
    u_sc = sc_decode(llr_ch, frozen_bits)
    return u, u_sc
