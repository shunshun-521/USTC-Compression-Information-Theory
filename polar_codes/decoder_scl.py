"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _prepare_channel_llr,
)
from utils import crc_encode, crc_check


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new_path = _Path(self.L.shape[0], self.L.shape[1] - 1, self.L[:, 0])
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, paths, l):
        for path in paths:
            for s in range(self.n - _active_llr_level(l, self.n), self.n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = f_operation(
                            path.L[j, s], path.L[j + branch_size, s]
                        )
                    else:
                        top_bit = int(path.B[j - branch_size, s + 1])
                        path.L[j, s + 1] = g_operation(
                            path.L[j - branch_size, s], path.L[j, s], top_bit
                        )

    def _propagate_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    )
                    path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            self._update_llrs(paths, l)
            new_paths = []

            if self.frozen_bits[l]:
                for path in paths:
                    np_ = path.copy()
                    llr = np_.L[l, self.n]
                    np_.pm += self._pm_penalty(llr, 0)
                    np_.B[l, self.n] = 0
                    np_.u_hat[l] = 0
                    self._propagate_bits(np_, l)
                    new_paths.append(np_)
            else:
                for path in paths:
                    llr = path.L[l, self.n]
                    for bit in (0, 1):
                        np_ = path.copy()
                        np_.pm += self._pm_penalty(llr, bit)
                        np_.B[l, self.n] = bit
                        np_.u_hat[l] = bit
                        self._propagate_bits(np_, l)
                        new_paths.append(np_)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = []
            for path in paths:
                if crc_check(path.u_hat[info_idx], self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
