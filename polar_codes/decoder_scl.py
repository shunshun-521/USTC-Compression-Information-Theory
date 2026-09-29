"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    sc_decode,
    _llr_channel_reorder,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)
from utils import crc_encode as _crc_encode_util
from utils import crc_check as _crc_check_util


def crc_encode(info_bits, crc_length=8):
    return _crc_encode_util(info_bits, crc_length)


def crc_check(bits, crc_length=8):
    return _crc_check_util(bits, crc_length)


class _Path:
    __slots__ = ("pm", "L", "B", "active")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1))
        self.L[:, 0] = llr_ch
        self.active = True


class SCLDecoder:
    """SCL 译码器（PSCD 结构 + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L_size = list_size
        self.crc_length = crc_length

    def _path_metric_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    l_bot = path.L[j, s]
                    l_top = path.L[j - branch_size, s]
                    b_top = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = l_bot + l_top if b_top == 0 else l_bot - l_top

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j - branch_size, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        if self.L_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = _llr_channel_reorder(llr_ch, self.N)
        paths = [_Path(self.N, self.n, llr_ch)]

        decode_order = [_bit_reversed(i, self.n) for i in range(self.N)]

        for l in decode_order:
            for p in paths:
                if p.active:
                    self._update_llrs(p, l)

            new_paths = []
            for p in paths:
                if not p.active:
                    continue
                llr = p.L[l, self.n]
                if l in self.frozen_set:
                    pen = self._path_metric_penalty(llr, 0)
                    p.pm += pen
                    p.B[l, self.n] = 0
                    self._update_bits(p, l)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        cp = _Path(self.N, self.n, llr_ch)
                        cp.pm = p.pm + self._path_metric_penalty(llr, u)
                        cp.L = p.L.copy()
                        cp.B = p.B.copy()
                        cp.B[l, self.n] = u
                        self._update_bits(cp, l)
                        new_paths.append(cp)

            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.L_size]

        candidates = []
        for p in paths:
            u_hat = p.B[:, self.n].astype(int)
            candidates.append((p.pm, u_hat))

        if self.crc_length > 0:
            info_idx = [i for i in range(self.N) if i not in self.frozen_set]
            info_idx.sort()
            valid = [(pm, u) for pm, u in candidates if crc_check(u[info_idx], self.crc_length)]
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][0]
