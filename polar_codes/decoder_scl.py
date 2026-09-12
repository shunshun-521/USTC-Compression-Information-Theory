"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Permuted SC
"""
import math

import numpy as np

from decoder_sc import _SCDCore, _prepare_llr, active_bit_level, active_llr_level, bit_reversed, lower_llr, upper_llr


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f'Unsupported CRC length: {crc_length}')


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([np.asarray(info_bits, dtype=int), np.array(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class _Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat', 'N', 'n')

    def __init__(self, N, n, llr_ch):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Permuted SC + Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def _copy_path(self, path):
        new_path = _Path(self.N, self.n, path.L[:, 0])
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr_val, u_val):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_val == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = _prepare_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    p = self._copy_path(path)
                    p.pm += self._pm_penalty(llr_val, 0)
                    p.B[l, self.n] = 0
                    p.u_hat[l] = 0
                    self._update_bits(p, l)
                    new_paths.append(p)
                else:
                    for u_val in (0, 1):
                        p = self._copy_path(path)
                        p.pm += self._pm_penalty(llr_val, u_val)
                        p.B[l, self.n] = u_val
                        p.u_hat[l] = u_val
                        self._update_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            crc_paths = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(crc_paths or paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat, best.pm
