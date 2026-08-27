"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
)
from encoder import bit_reversed


CRC8_POLY = [1, 0, 0, 0, 0, 0, 1, 1, 1]
CRC16_POLY = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _crc_mod2(msg, gen):
    L = len(gen) - 1
    reg = [int(b) for b in msg] + [0] * L
    for i in range(len(msg)):
        if reg[i] == 1:
            for j in range(len(gen)):
                reg[i + j] ^= gen[j]
    return np.array(reg[len(msg) :], dtype=np.int8)


def _crc_check_mod2(bits, gen):
    L = len(gen) - 1
    reg = [int(b) for b in bits]
    for i in range(len(bits) - L):
        if reg[i] == 1:
            for j in range(len(gen)):
                reg[i + j] ^= gen[j]
    return all(x == 0 for x in reg[len(bits) - L :])


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    gen = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return np.concatenate([info_bits, _crc_mod2(info_bits, gen)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    gen = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_check_mod2(bits, gen)


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        child = _Path(self.L.shape[0], self.L.shape[1] - 1)
        child.L = self.L.copy()
        child.B = self.B.copy()
        child.pm = self.pm
        child.u_hat = self.u_hat.copy()
        return child


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_phi = path.L[l, self.n]

                if self.frozen_bits[l]:
                    path.pm = _pm_update(path.pm, llr_phi, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    candidates.append((path.pm, path))
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm = _pm_update(child.pm, llr_phi, u)
                        child.u_hat[l] = u
                        child.B[l, self.n] = u
                        self._update_bits(child, l)
                        candidates.append((child.pm, child))

            candidates.sort(key=lambda x: x[0])
            paths = [c[1] for c in candidates[: self.list_size]]

        best = self._select_best_path(paths)
        return best.u_hat, best.pm

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                return min(valid, key=lambda p: p.pm)
        return min(paths, key=lambda p: p.pm)
