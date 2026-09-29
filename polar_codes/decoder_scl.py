"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于置换 SC 树结构
"""
import math

import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    lower_llr,
    upper_llr,
)
from encoder import bit_reversal_int


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in info_bits:
        fb = ((reg >> (crc_length - 1)) ^ int(b)) & 1
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        fb = ((reg >> (crc_length - 1)) ^ int(b)) & 1
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    return reg == 0


class _PathState:
    __slots__ = ("L", "B", "pm", "u_hat", "copied")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1))
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.copied = False


class SCLDecoder:
    """SCL 译码器（Lazy Copy + 置换 SC）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.phase_order = [bit_reversal_int(i, self.n) for i in range(N)]

    def _duplicate(self, path):
        new_path = _PathState(self.N, self.n)
        new_path.L[:] = path.L
        new_path.B[:] = path.B
        new_path.pm = path.pm
        new_path.u_hat[:] = path.u_hat
        new_path.copied = True
        return new_path

    def _branch(self, path, reuse):
        if reuse and not path.copied:
            path.copied = True
            return path
        return self._duplicate(path)

    @staticmethod
    def _metric_penalty(llr, u):
        u_pref = 0 if llr >= 0 else 1
        return 0.0 if u == u_pref else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = lower_llr(path.L[j, s], path.B[j - branch_size, s], int(top_bit))

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        root = _PathState(self.N, self.n)
        root.L[:, 0] = llr_ch
        paths = [root]

        for l in self.phase_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    new_path = self._branch(path, reuse=True)
                    new_path.pm += self._metric_penalty(llr, 0)
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = self._duplicate(path)
                        new_path.pm += self._metric_penalty(llr, u)
                        new_path.B[l, self.n] = u
                        new_path.u_hat[l] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
