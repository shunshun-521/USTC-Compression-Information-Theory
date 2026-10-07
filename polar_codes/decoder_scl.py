"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    bit_reversed,
    upper_llr,
    lower_llr,
    active_llr_level,
    active_bit_level,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) & 1) ^ int(bit)
        reg = ((reg << 1) & mask) ^ (poly if fb else 0)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return _crc_remainder(bits, crc_length) == 0


class _Path:
    __slots__ = ("parent", "L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr, parent=None):
        if parent is None:
            self.parent = None
            self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
            self.B = np.full((N, n + 1), np.nan)
            self.L[:, 0] = llr
            self.pm = 0.0
            self.u_hat = np.zeros(N, dtype=np.int8)
        else:
            self.parent = parent
            self.L = parent.L
            self.B = parent.B.copy()
            self.pm = parent.pm
            self.u_hat = parent.u_hat.copy()


class SCLDecoder:
    """SCL 译码器（Lazy Copy：分裂时仅复制 B 与 u_hat）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        L, B = path.L, path.B
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        B = path.B
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        rev = bit_reversal_permutation(self.N)
        llr = np.asarray(llr_ch, dtype=np.float64)[rev]

        paths = [_Path(self.N, self.n, llr)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path.L[l, self.n]
                if l in self.frozen_set:
                    pen = self._pm_penalty(llr_bit, 0)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N, self.n, llr, parent=path)
                        child.pm += self._pm_penalty(llr_bit, bit)
                        child.B[l, self.n] = bit
                        child.u_hat[l] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best_crc = None
        best_pm = None
        for path in paths:
            u = path.u_hat.astype(int)
            if self.crc_length > 0:
                info_bits = u[self.info_positions]
                if not crc_check(info_bits, self.crc_length):
                    continue
            if best_pm is None or path.pm < best_pm:
                best_pm = path.pm
                best_crc = u

        if best_crc is not None:
            return best_crc, best_pm

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
