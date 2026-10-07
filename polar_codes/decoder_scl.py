"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed,
    active_llr_level,
    active_bit_level,
    upper_llr,
    lower_llr,
    sc_decode,
)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = 0x07 if crc_length == 8 else 0x8005
    reg = 0
    mask = (1 << crc_length) - 1
    for b in info_bits:
        fb = ((reg >> (crc_length - 1)) ^ int(b)) & 1
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        N, n = self.N, self.n
        L, B = path.L, path.B
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def _update_bits(self, path, l):
        N, n = self.N, self.n
        B = path.B
        if l < N / 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _penalty(self, llr_val, bit):
        if (bit == 0 and llr_val >= 0) or (bit == 1 and llr_val < 0):
            return 0.0
        return abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_leaf = path.L[l, self.n]
                if l in self.frozen_set:
                    bit = 0
                    path.pm += self._penalty(llr_leaf, bit)
                    path.B[l, self.n] = bit
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        cp = Path(self.N, self.n, llr_ch)
                        cp.pm = path.pm + self._penalty(llr_leaf, bit)
                        cp.L[:] = path.L
                        cp.B[:] = path.B
                        cp.B[l, self.n] = bit
                        self._update_bits(cp, l)
                        new_paths.append(cp)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.B[:, self.n].astype(np.int8)[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            chosen = min(valid or paths, key=lambda p: p.pm)
        else:
            chosen = min(paths, key=lambda p: p.pm)

        return chosen.B[:, self.n].astype(np.int8), chosen.pm


def scl_equivalent_to_sc(N, frozen_bits, llr_ch):
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0).decode(llr_ch)
    u_sc = sc_decode(llr_ch, frozen_bits)
    return np.array_equal(u_scl, u_sc)
