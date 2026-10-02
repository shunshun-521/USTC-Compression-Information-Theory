"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import copy
from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    from utils import crc_encode as _ce

    return _ce(info_bits, crc_length)


def crc_check(bits, crc_length=8):
    from utils import crc_check as _cc

    return _cc(bits, crc_length)


class _Path:
    __slots__ = ("L", "B", "pm", "n", "N")

    def __init__(self, llr_ch, n, N):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.pm = 0.0

    def copy(self):
        p = _Path(np.zeros(self.N), self.n, self.N)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        return p

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = int(self.B[j - branch_size, s + 1])
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j - branch_size, s], self.L[j, s], top_bit
                    )

    def _update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    @staticmethod
    def _pm_add(pm, llr, u):
        v = 0 if llr >= 0 else 1
        return pm + (0.0 if u == v else abs(llr))


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(llr_ch, self.n, self.N)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                path._update_llrs(l)
                llr = path.L[l, self.n]
                if self.frozen_bits[l]:
                    p = path.copy()
                    p.pm = _Path._pm_add(path.pm, llr, 0)
                    p.B[l, self.n] = 0
                    p._update_bits(l)
                    candidates.append(p)
                else:
                    for u in (0, 1):
                        p = path.copy()
                        p.pm = _Path._pm_add(path.pm, llr, u)
                        p.B[l, self.n] = u
                        p._update_bits(l)
                        candidates.append(p)
            candidates.sort(key=lambda x: x.pm)
            paths = candidates[: self.list_size]

        u_hat = paths[0].B[:, self.n].astype(int)
        best_pm = paths[0].pm

        if self.crc_length > 0:
            for p in paths:
                u_try = p.B[:, self.n].astype(int)
                bits = u_try[self.info_indices]
                if crc_check(bits, self.crc_length):
                    return u_try.copy(), p.pm
        return paths[0].B[:, self.n].astype(int), paths[0].pm


def bits_to_u(path, info_indices, N):
    return path.B[:, path.n].astype(int)
