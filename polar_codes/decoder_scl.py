"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _update_llrs,
    _update_bits,
    _frozen_set,
    bit_reversed,
    active_llr_level,
    active_bit_level,
)


from crc_utils import crc_encode, crc_check


class PathState:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, n, N):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen = _frozen_set(frozen_bits)
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _llr_penalty(llr_val, u):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u == hard else abs(llr_val)

    def _copy_path(self, src):
        dst = PathState(self.n, self.N)
        dst.pm = src.pm
        dst.u_hat = src.u_hat.copy()
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        return dst

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        paths = [PathState(n, N)]
        paths[0].L[:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n)
                llr_val = path.L[l, n]

                if l in self.frozen:
                    path.pm += self._llr_penalty(llr_val, 0)
                    path.u_hat[l] = 0
                    path.B[l, n] = 0
                    _update_bits(path.B, l, n)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        p = self._copy_path(path)
                        p.pm += self._llr_penalty(llr_val, u)
                        p.u_hat[l] = u
                        p.B[l, n] = u
                        _update_bits(p.B, l, n)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
