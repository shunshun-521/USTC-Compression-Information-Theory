"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import (
    _SCDCore,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
)
from encoder import bit_reversal_permutation


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
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, crc_bits]).astype(int)


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特表）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _init_paths(self, llr_ch):
        p = _Path(self.N, self.n)
        p.L[:, 0] = llr_ch[self.br]
        return [p]

    def _update_llrs_path(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits_path(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._init_paths(llr_ch)

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs_path(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen:
                    penalty = abs(llr) if llr < 0 else 0.0
                    new = self._clone_path(path)
                    new.pm += penalty
                    new.u_hat[l] = 0
                    new.B[l, self.n] = 0
                    self._update_bits_path(new, l)
                    candidates.append(new)
                else:
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0) else abs(llr)
                        new = self._clone_path(path)
                        new.pm += penalty
                        new.u_hat[l] = bit
                        new.B[l, self.n] = bit
                        self._update_bits_path(new, l)
                        candidates.append(new)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            info_idx = sorted(set(range(self.N)) - self.frozen)
            valid = [p for p in paths if crc_check(p.u_hat[info_idx], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm

    def _clone_path(self, path):
        new = _Path(self.N, self.n)
        new.L = path.L.copy()
        new.B = path.B.copy()
        new.pm = path.pm
        new.u_hat = path.u_hat.copy()
        return new
