"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助 CA-SCL
基于 Permuted SC 的列表扩展
"""
import math
import numpy as np
from decoder_sc import (
    _PermutedSCD,
    _bit_reversed_int,
    active_llr_level,
    active_bit_level,
    upper_llr,
    lower_llr,
    _hard_decision,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    data = bits[:-crc_length]
    expected = crc_encode(data, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs(self, paths, l):
        for p in paths:
            for s in range(self.n - active_llr_level(l, self.n), self.n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        p.L[j, s + 1] = upper_llr(p.L[j, s], p.L[j + branch_size, s])
                    else:
                        p.L[j, s + 1] = lower_llr(
                            p.L[j, s],
                            p.L[j - branch_size, s],
                            int(p.B[j - branch_size, s + 1]),
                        )

    def _update_bits(self, paths, l):
        if l < self.N / 2:
            return
        for p in paths:
            for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        p.B[j - branch_size, s - 1] = int(p.B[j, s]) ^ int(
                            p.B[j - branch_size, s]
                        )
                        p.B[j, s - 1] = p.B[j, s]

    def _pm_penalty(self, llr, u):
        u_hat = 0 if llr >= 0 else 1
        return 0.0 if u_hat == u else abs(llr)

    def decode(self, llr_ch):
        from encoder import bit_reversal_permutation

        llr = np.asarray(llr_ch, dtype=np.float64)[bit_reversal_permutation(self.N)]
        paths = [_Path(self.N, self.n, llr.copy())]

        for i in range(self.N):
            l = _bit_reversed_int(i, self.n)
            self._update_llrs(paths, l)
            new_paths = []
            for p in paths:
                cur_llr = p.L[l, self.n]
                if l in self.frozen_set:
                    pen = self._pm_penalty(cur_llr, 0)
                    p.pm += pen
                    p.u_hat[l] = 0
                    p.B[l, self.n] = 0
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        cp = _Path(self.N, self.n, p.L[:, 0].copy())
                        cp.L[:, 1:] = p.L[:, 1:].copy()
                        cp.B[:, :] = p.B[:, :].copy()
                        cp.pm = p.pm + self._pm_penalty(cur_llr, u)
                        cp.u_hat = p.u_hat.copy()
                        cp.u_hat[l] = u
                        cp.B[l, self.n] = u
                        new_paths.append(cp)
            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.L_size]
            self._update_bits(paths, l)

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            if valid:
                paths = valid
        best = min(paths, key=lambda x: x.pm)
        return best.u_hat.astype(int), best.pm
