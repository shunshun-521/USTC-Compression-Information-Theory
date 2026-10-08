"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _frozen_set_from_mask,
    _lower_llr,
    _upper_llr,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.uint8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.uint8)
    if crc_length == 0:
        return True
    return np.all(crc_encode(bits[:-crc_length], crc_length)[-crc_length:] == bits[-crc_length:])


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = _frozen_set_from_mask(frozen_bits)
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.perm = bit_reversal_permutation(N)

    def _pm_add(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u == u_hard:
            return pm
        return pm + abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.perm]
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_i = path.L[l, self.n]
                if l in self.frozen_set:
                    pm = self._pm_add(path.pm, llr_i, 0)
                    child = self._copy_path(path)
                    child.pm = pm
                    child.B[l, self.n] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        pm = self._pm_add(path.pm, llr_i, u)
                        child = self._copy_path(path)
                        child.pm = pm
                        child.B[l, self.n] = u
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = paths[0]
        u_hat = best.B[:, self.n].astype(int)
        best_pm = best.pm

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            for p in sorted(paths, key=lambda x: x.pm):
                cand = p.B[:, self.n].astype(int)
                payload = cand[info_idx]
                if crc_check(payload, self.crc_length):
                    return cand, p.pm
        return u_hat, best_pm

    def _copy_path(self, path):
        q = _Path(self.N, self.n)
        q.pm = path.pm
        q.L = path.L.copy()
        q.B = path.B.copy()
        return q

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = 0 if np.isnan(path.B[j - branch_size, s + 1]) else int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = _lower_llr(path.L[j, s], path.L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]
