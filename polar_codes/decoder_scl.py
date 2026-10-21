"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Permuted SC 树
"""
import numpy as np
from decoder_sc import (
    bit_reversed,
    active_llr_level,
    active_bit_level,
    _upper_llr,
    _lower_llr,
)
from utils_crc import crc_encode, crc_check


class _PathState:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制 + PM 剪枝）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=int) if info_indices is not None else None
        )

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
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
    def _pm_penalty(llr, u):
        u_llr = 0 if llr >= 0 else 1
        return 0.0 if u == u_llr else abs(llr)

    def _copy_path(self, path):
        new = _PathState(self.N, self.n)
        new.pm = path.pm
        new.L[:] = path.L
        new.B[:] = path.B
        new.u_hat[:] = path.u_hat
        return new

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    new_p = self._copy_path(path)
                    new_p.pm += self._pm_penalty(llr, 0)
                    new_p.B[l, self.n] = 0
                    new_p.u_hat[l] = 0
                    self._update_bits(new_p, l)
                    candidates.append(new_p)
                else:
                    for u in (0, 1):
                        new_p = self._copy_path(path)
                        new_p.pm += self._pm_penalty(llr, u)
                        new_p.B[l, self.n] = u
                        new_p.u_hat[l] = u
                        self._update_bits(new_p, l)
                        candidates.append(new_p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = (
                    p.u_hat[self.info_indices]
                    if self.info_indices is not None
                    else p.u_hat
                )
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm


__all__ = ["SCLDecoder", "crc_encode", "crc_check"]
