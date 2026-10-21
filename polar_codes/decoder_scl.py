"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
    channel_llr_for_decoder,
    _frozen_index_set,
)
from utils import crc_encode, crc_check


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        p = _Path.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Lazy Copy：分裂时复制路径对象）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = _frozen_index_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(
            [i for i in range(N) if i not in self.frozen_set], dtype=np.int64
        )

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s], path.L[j - branch, s], path.B[j - branch, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch:
                    path.B[j - branch, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr = channel_llr_for_decoder(llr_ch)
        paths = [_Path(self.N, self.n, llr)]

        decode_order = [_bit_reversed(i, self.n) for i in range(self.N)]

        for l in decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(cur_llr, 0)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for u_bit in (0, 1):
                        new_path = path.copy()
                        pen = self._pm_penalty(cur_llr, u_bit)
                        new_path.pm += pen
                        new_path.B[l, self.n] = u_bit
                        new_path.u_hat[l] = u_bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_pm = paths[0].pm
        best_u = paths[0].u_hat

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                valid.sort(key=lambda p: p.pm)
                best_u = valid[0].u_hat
                best_pm = valid[0].pm
            else:
                paths.sort(key=lambda p: p.pm)
                best_u = paths[0].u_hat
                best_pm = paths[0].pm
        else:
            paths.sort(key=lambda p: p.pm)
            best_u = paths[0].u_hat
            best_pm = paths[0].pm

        return best_u, best_pm
