"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level
from utils import crc_encode as utils_crc_encode, crc_check as utils_crc_check


def crc_encode(info_bits, crc_length=8):
    return utils_crc_encode(info_bits, crc_length)


def crc_check(bits, crc_length=8):
    return utils_crc_check(bits, crc_length)


def _llr_to_decoder_order(llr_ch):
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv = np.argsort(br)
    return np.asarray(llr_ch, dtype=np.float64)[inv]


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（路径复制实现，L 较小时足够高效）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _bit_llr(self, path, l):
        L, B = path.L, path.B
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )
        return L[l, n]

    def _update_bits(self, path, l):
        B = path.B
        n = self.n
        if l < self.N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0.0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = _llr_to_decoder_order(llr_ch)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = int(self.br[i])
            candidates = []

            for path in paths:
                llr_val = self._bit_llr(path, l)
                if self.frozen_bits[l]:
                    u = 0
                    new_pm = path.pm + self._pm_penalty(llr_val, u)
                    candidates.append((new_pm, path, u))
                else:
                    for u in (0, 1):
                        new_pm = path.pm + self._pm_penalty(llr_val, u)
                        candidates.append((new_pm, path, u))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for pm, parent, u in selected:
                child = _Path(self.N, self.n, llr_ch)
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()
                child.pm = pm
                child.B[l, self.n] = u
                child.u_hat[l] = u
                self._update_bits(child, l)
                new_paths.append(child)
            paths = new_paths

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            info_idx = np.where(self.frozen_bits == 0)[0]
            valid = [
                p for p in paths if crc_check(p.u_hat[info_idx], self.crc_length)
            ]
            if valid:
                paths = valid

        best = paths[0]
        return best.u_hat.astype(int), best.pm
