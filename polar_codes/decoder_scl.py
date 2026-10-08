"""
极化码 SCL（串行抵消列表）译码器
基于 Permuted SC 因子图，支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    bit_reversed,
    active_llr_level,
    active_bit_level,
    upper_llr,
    lower_llr,
    sc_decode,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & mask) ^ (poly if fb else 0)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    reg = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if len(bits) < crc_length:
        return False
    return _crc_remainder(bits, crc_length) == 0


class _PathState:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch

    def copy(self):
        p = _PathState.__new__(_PathState)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        return p


class SCLDecoder:
    """SCL 译码器（路径复制 Lazy Copy：仅在分裂时 copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path.L[l, self.n]
                if l in self.frozen_set:
                    pen = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    cp = path.copy()
                    cp.pm += pen
                    cp.B[l, self.n] = 0
                    self._update_bits(cp, l)
                    new_paths.append(cp)
                else:
                    for bit in (0, 1):
                        pen = 0.0 if (llr_bit >= 0 and bit == 0) or (llr_bit < 0 and bit == 1) else abs(llr_bit)
                        cp = path.copy()
                        cp.pm += pen
                        cp.B[l, self.n] = bit
                        self._update_bits(cp, l)
                        new_paths.append(cp)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best_pm = min(paths, key=lambda p: p.pm)
        best_crc = None
        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            for p in paths:
                payload = p.B[:, self.n].astype(int)
                if crc_check(payload[info_mask], self.crc_length):
                    if best_crc is None or p.pm < best_crc.pm:
                        best_crc = p

        chosen = best_crc if best_crc is not None else best_pm
        u_hat = chosen.B[:, self.n].astype(int)
        return u_hat, chosen.pm


def verify_scl_equals_sc(N=64, frozen_bits=None, llr=None):
    from construction import ga_construction

    if frozen_bits is None:
        info_idx, _, _ = ga_construction(N, N // 2, 2.5)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
    if llr is None:
        rng = np.random.default_rng(1)
        llr = rng.normal(0, 2, size=N)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    return np.array_equal(u_sc, u_scl)
