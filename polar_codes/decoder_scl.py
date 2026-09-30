"""
极化码 SCL（串行抵消列表）译码器
Permuted SCD 框架 + CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed_int,
    active_llr_level,
    active_bit_level,
    _update_llrs,
    _update_bits,
    precompute_sc_indices,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if len(bits) < crc_length:
        return False
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Permuted SCD）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        precompute_sc_indices(N)

    @staticmethod
    def _penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed_int(i, self.n)
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr = path.L[l, self.n]
                if llr == 0.0 and l not in self.frozen_set:
                    llr = llr_ch[l]
                if l in self.frozen_set:
                    p = _Path(self.N, self.n, llr_ch)
                    p.L[:] = path.L
                    p.B[:] = path.B
                    p.pm = path.pm + self._penalty(llr, 0)
                    p.u_hat[:] = path.u_hat
                    p.B[l, self.n] = 0
                    p.u_hat[l] = 0
                    _update_bits(p.B, l, self.n, self.N)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = _Path(self.N, self.n, llr_ch)
                        p.L[:] = path.L
                        p.B[:] = path.B
                        p.pm = path.pm + self._penalty(llr, u)
                        p.u_hat[:] = path.u_hat
                        p.B[l, self.n] = u
                        p.u_hat[l] = u
                        _update_bits(p.B, l, self.n, self.N)
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.L_size]

        if self.crc_length > 0:
            ok = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(ok if ok else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
