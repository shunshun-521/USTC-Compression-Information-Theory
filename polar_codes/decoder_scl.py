"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


CRC8_POLY_BITS = [1, 0, 0, 0, 0, 0, 1, 1, 1]
CRC16_POLY_BITS = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _crc_remainder_mod2(msg, poly_bits):
    msg = [int(b) for b in msg]
    r = len(poly_bits) - 1
    reg = msg + [0] * r
    for i in range(len(msg)):
        if reg[i] == 1:
            for j in range(len(poly_bits)):
                reg[i + j] ^= poly_bits[j]
    return reg[-r:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY_BITS if crc_length == 8 else CRC16_POLY_BITS
    remainder = _crc_remainder_mod2(info_bits, poly)
    return np.concatenate([info_bits, np.array(remainder, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    poly = CRC8_POLY_BITS if crc_length == 8 else CRC16_POLY_BITS
    remainder = _crc_remainder_mod2(bits, poly)
    return all(x == 0 for x in remainder)


class PathState:
    """单条 SCL 路径状态。"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new = PathState(self.L.shape[0], self.L.shape[1] - 1)
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    """SCL 译码器（Permuted SCD + Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.rev = bit_reversal_permutation(N)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j, s], path.L[j - branch_size, s], path.B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)[self.rev]

        paths = [PathState(self.N, self.n)]
        paths[0].L[:, 0] = llr

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)

            for path in paths:
                self._update_llrs(path, l)

            if l in self.frozen_set:
                for path in paths:
                    path.pm += self._pm_penalty(path.L[l, self.n], 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
            else:
                new_paths = []
                for path in paths:
                    llr_val = path.L[l, self.n]
                    for u in (0, 1):
                        child = path.copy()
                        child.pm += self._pm_penalty(llr_val, u)
                        child.u_hat[l] = u
                        child.B[l, self.n] = u
                        new_paths.append(child)
                new_paths.sort(key=lambda p: p.pm)
                paths = new_paths[: self.list_size]

            for path in paths:
                self._update_bits(path, l)

        best_crc_path = None
        best_crc_pm = float("inf")
        best_path = min(paths, key=lambda p: p.pm)

        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length) and path.pm < best_crc_pm:
                    best_crc_pm = path.pm
                    best_crc_path = path

        chosen = best_crc_path if best_crc_path is not None else best_path
        return chosen.u_hat.copy(), chosen.pm
