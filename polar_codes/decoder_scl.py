"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助 CA-SCL
"""
import numpy as np
import math

from decoder_sc import f_operation, g_operation, _bit_reversed, _active_llr_level, _active_bit_level
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
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（路径复制实现，列表规模适中时足够高效）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen)[0]

    def _update_llrs(self, paths, l):
        for p in paths:
            for s in range(self.n - _active_llr_level(l, self.n), self.n):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        p.L[j, s + 1] = f_operation(p.L[j, s], p.L[j + branch_size, s])
                    else:
                        btm = p.L[j, s]
                        top = p.L[j - branch_size, s]
                        bit = p.B[j - branch_size, s + 1]
                        p.L[j, s + 1] = g_operation(btm, top, bit)

    def _update_bits(self, paths, l):
        if l < self.N / 2:
            return
        for p in paths:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        p.B[j - branch_size, s - 1] = (
                            int(p.B[j, s]) ^ int(p.B[j - branch_size, s])
                        )
                        p.B[j, s - 1] = p.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        root = Path(self.N, self.n)
        root.L[:, 0] = llr_ch
        paths = [root]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            self._update_llrs(paths, l)

            new_paths = []
            for p in paths:
                llr = p.L[l, self.n]
                if self.frozen[l]:
                    pen = self._pm_penalty(llr, 0)
                    p.pm += pen
                    p.B[l, self.n] = 0
                    p.u_hat[l] = 0
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        cp = Path(self.N, self.n)
                        cp.L[...] = p.L
                        cp.B[...] = p.B
                        cp.u_hat[...] = p.u_hat
                        cp.pm = p.pm + self._pm_penalty(llr, bit)
                        cp.B[l, self.n] = bit
                        cp.u_hat[l] = bit
                        new_paths.append(cp)

            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.L_size]

            for p in paths:
                self._update_bits([p], l)

        u_best, pm_best = paths[0].u_hat.astype(int), paths[0].pm

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            if valid:
                valid.sort(key=lambda x: x.pm)
                u_best = valid[0].u_hat.astype(int)
                pm_best = valid[0].pm

        return u_best, pm_best
