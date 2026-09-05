"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
)
from utils import crc_check, crc_encode

__all__ = ["SCLDecoder", "crc_encode", "crc_check"]


class _PathState:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int32)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _copy_path(self, path):
        new_path = _PathState(self.N, self.n, path.L[:, 0])
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.N, self.n, llr_ch)]

        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            candidates = []
            is_frozen = l in self.frozen_set

            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path.L[l, self.n]

                if is_frozen:
                    new_path = self._copy_path(path)
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    new_path.pm += self._pm_penalty(llr_bit, 0)
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        new_path.pm += self._pm_penalty(llr_bit, bit)
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = paths[0]
        best_pm = best.pm
        crc_best = None

        for path in paths:
            if self.crc_length > 0:
                payload = path.u_hat[self.info_indices]
                if crc_check(payload, self.crc_length) and path.pm < best_pm:
                    crc_best = path
                    best_pm = path.pm
            elif path.pm < best_pm:
                best = path
                best_pm = path.pm

        if crc_best is not None:
            best = crc_best

        return best.u_hat.copy(), best.pm
