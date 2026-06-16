"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
)
from encoder import bit_reversal_permutation
from utils import crc_check_bits, crc_encode, crc_check

__all__ = ["SCLDecoder", "crc_encode", "crc_check"]


class Path:
  __slots__ = ("L", "B", "pm", "active")

  def __init__(self, N, n, llr_ch):
    self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
    self.B = np.full((N, n + 1), np.nan)
    self.L[:, 0] = llr_ch
    self.pm = 0.0
    self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 Path 对象）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.br = bit_reversal_permutation(N)
        self.frozen_br = self.frozen_bits[self.br]
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l, u_val):
        path.B[l, self.n] = u_val
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]
        u_br_paths = [np.zeros(self.N, dtype=int)]

        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            for p in paths:
                self._update_llrs(p, l)

            new_paths = []
            new_u = []

            if self.frozen_br[l]:
                for p, u_br in zip(paths, u_br_paths):
                    llr = p.L[l, self.n]
                    u_val = 0
                    p.pm += self._path_metric_penalty(llr, u_val)
                    self._update_bits(p, l, u_val)
                    u_br = u_br.copy()
                    u_br[l] = u_val
                    new_paths.append(p)
                    new_u.append(u_br)
            else:
                for p, u_br in zip(paths, u_br_paths):
                    llr = p.L[l, self.n]
                    for u_val in (0, 1):
                        cp = Path(self.N, self.n, llr_ch)
                        cp.L = p.L.copy()
                        cp.B = p.B.copy()
                        cp.pm = p.pm + self._path_metric_penalty(llr, u_val)
                        self._update_bits(cp, l, u_val)
                        u_copy = u_br.copy()
                        u_copy[l] = u_val
                        new_paths.append(cp)
                        new_u.append(u_copy)

            order = np.argsort([p.pm for p in new_paths])
            paths = [new_paths[i] for i in order[: self.list_size]]
            u_br_paths = [new_u[i] for i in order[: self.list_size]]

        candidates = []
        for p, u_br in zip(paths, u_br_paths):
            u_hat = np.empty(self.N, dtype=int)
            u_hat[self.br] = u_br
            candidates.append((p.pm, u_hat))

        if self.crc_length > 0:
            valid = [
                (pm, u)
                for pm, u in candidates
                if crc_check_bits(u[self.info_positions], self.crc_length)
            ]
            if valid:
                candidates = valid

        best_pm, best_u = min(candidates, key=lambda x: x[0])
        return best_u, best_pm
