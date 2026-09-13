"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from crc_utils import crc_check, crc_encode
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _as_frozen_mask,
    _bit_reversed,
    _prepare_llr,
    _update_bits,
    _update_llrs,
)


def _path_metric_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = _as_frozen_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _clone(self, path):
        new = _Path(self.N, self.n, path.L[:, 0])
        new.L = path.L.copy()
        new.B = path.B.copy()
        new.pm = path.pm
        new.u_hat = path.u_hat.copy()
        return new

    def decode(self, llr_ch):
        llr = _prepare_llr(llr_ch)
        N, n = self.N, self.n
        paths = [_Path(N, n, llr)]

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n)
                llr_bit = path.L[l, n]

                if self.frozen_bits[l]:
                    child = self._clone(path)
                    child.pm += _path_metric_penalty(llr_bit, 0)
                    child.u_hat[l] = 0
                    child.B[l, n] = 0
                    _update_bits(child.B, l, n, N)
                    candidates.append(child)
                else:
                    for u_bit in (0, 1):
                        child = self._clone(path)
                        child.pm += _path_metric_penalty(llr_bit, u_bit)
                        child.u_hat[l] = u_bit
                        child.B[l, n] = u_bit
                        _update_bits(child.B, l, n, N)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
