"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    f_operation,
    g_operation,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
        width = 8
    elif crc_length == 16:
        poly = 0x8005
        width = 16
    else:
        raise ValueError("crc_length must be 8 or 16")

    mask = (1 << width) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(8):
            if reg & (1 << (width - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask

    crc_bits = np.array([(reg >> (width - 1 - i)) & 1 for i in range(width)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    recomputed = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(recomputed, bits)


class Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = Path(len(self.L), int(math.log2(len(self.L))))
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _pm_update(self, pm, llr, u):
        v = 0 if llr >= 0 else 1
        if u != v:
            pm += abs(llr)
        return pm

    def _continue_path(self, path, l, llr_val, u_bit):
        path.pm = self._pm_update(path.pm, llr_val, u_bit)
        path.B[l, self.n] = u_bit
        path.u_hat[l] = u_bit

        if l >= self.N // 2:
            for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                            path.B[j - branch_size, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    def _compute_llr(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    btm = path.L[j, s]
                    top = path.L[j - branch_size, s]
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(btm, top, top_bit)
        return path.L[l, self.n]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = llr_ch[self.br]

        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_internal

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                llr_val = self._compute_llr(path, l)
                if l in self.frozen_set:
                    p0 = path.copy()
                    self._continue_path(p0, l, llr_val, 0)
                    new_paths.append(p0)
                else:
                    for u_bit in (0, 1):
                        p = path.copy()
                        self._continue_path(p, l, llr_val, u_bit)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid or paths, key=lambda p: p.pm)
        else:
            best = paths[0]

        u_hat = best.u_hat.copy()
        u_hat[self.frozen_bits] = 0
        return u_hat, best.pm
