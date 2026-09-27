"""
极化码 SCL（串行抵消列表）译码器，含 CRC 辅助 CA-SCL
"""
import copy
import numpy as np

from decoder_sc import (
    bit_reversed,
    upper_llr,
    lower_llr,
    active_llr_level,
    active_bit_level,
    _channel_llr_to_decoder,
)
from utils import crc_encode, crc_check


class _PathState:
    __slots__ = ("L", "B", "pm", "u")

    def __init__(self, N, n, llr_dec):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_dec
        self.pm = 0.0
        self.u = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制实现，适用于中等 N/L）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs_path(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits_path(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _pm_penalty(llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_dec = _channel_llr_to_decoder(llr_ch)
        paths = [_PathState(self.N, self.n, llr_dec)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                self._update_llrs_path(path, l)
                llr_bit = path.L[l, self.n]
                if l in self.frozen_set:
                    pen = self._pm_penalty(llr_bit, 0)
                    path.pm += pen
                    path.u[l] = 0
                    path.B[l, self.n] = 0
                    self._update_bits_path(path, l)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        p = copy.deepcopy(path)
                        p.pm += self._pm_penalty(llr_bit, u_bit)
                        p.u[l] = u_bit
                        p.B[l, self.n] = u_bit
                        self._update_bits_path(p, l)
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u, self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = paths[0]
        else:
            best = paths[0]
        return best.u.copy(), best.pm
