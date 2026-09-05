"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    bit_reversed,
    active_llr_level,
    active_bit_level,
    upper_llr,
    lower_llr,
    sc_path_metric,
    _init_llr_matrix,
)


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_polynomial(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class PathState:
    __slots__ = ("L", "B", "pm", "u_hat", "N", "n")

    def __init__(self, llr_ch, N, n):
        self.N = N
        self.n = n
        self.L = _init_llr_matrix(llr_ch, N, n)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new_path = object.__new__(PathState)
        new_path.N = self.N
        new_path.n = self.n
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


def _advance_phase(path, l, frozen_bits):
    n = path.n
    N = path.N
    L = path.L
    B = path.B

    start = n - active_llr_level(l, n)
    for s in range(start, n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                if np.isnan(B[j - branch_size, s + 1]):
                    B[j - branch_size, s + 1] = 0.0
                L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1])

    llr_val = L[l, n]
    if frozen_bits[l]:
        u_bit = 0
        path.pm += sc_path_metric(llr_val, 0)
    else:
        u_bit = None

    return llr_val, u_bit


def _commit_phase(path, l, u_bit):
    path.u_hat[l] = u_bit
    path.B[l, path.n] = u_bit
    n = path.n
    N = path.N
    if l < N // 2:
        return
    stop = n - active_bit_level(l, n)
    for s in range(n, stop, -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                top = 0.0 if np.isnan(path.B[j, s]) else path.B[j, s]
                bottom = 0.0 if np.isnan(path.B[j - branch_size, s]) else path.B[j - branch_size, s]
                path.B[j - branch_size, s - 1] = int(top) ^ int(bottom)
                path.B[j, s - 1] = top


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.phase_order = [bit_reversed(i, self.n) for i in range(N)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [PathState(llr_ch, self.N, self.n)]

        for l in self.phase_order:
            candidates = []
            for path in paths:
                llr_val, forced = _advance_phase(path, l, self.frozen_bits)
                if forced is not None:
                    new_path = path.copy()
                    _commit_phase(new_path, l, forced)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = path.copy()
                        new_path.pm += sc_path_metric(llr_val, u_bit)
                        _commit_phase(new_path, l, u_bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = valid[0] if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
