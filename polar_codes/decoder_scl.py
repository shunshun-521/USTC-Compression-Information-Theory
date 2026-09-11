"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _lower_llr,
    _upper_llr,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


def _crc_remainder(bits, crc_length):
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    poly_shifted = poly if crc_length == 8 else poly >> (16 - crc_length)

    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly_shifted) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。

    使用标准多项式：
      r=8:  CRC-8  (0x07, 即 x^8 + x^2 + x + 1)
      r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    remainder = _crc_remainder(padded, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    返回 True/False。
    """
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return bool(np.array_equal(expected, bits))


class _SCLPath:
    """单条 SCL 路径，使用 lazy copy 共享 LLR/比特数组。"""

    __slots__ = ("L", "B", "pm", "parent_L", "parent_B", "copy_L", "copy_B")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.parent_L = None
        self.parent_B = None
        self.copy_L = False
        self.copy_B = False

    def ensure_llr_copy(self):
        if self.parent_L is not None and not self.copy_L:
            self.L = self.parent_L.L.copy()
            self.copy_L = True

    def ensure_bit_copy(self):
        if self.parent_B is not None and not self.copy_B:
            self.B = self.parent_B.B.copy()
            self.copy_B = True

    def fork(self):
        child = _SCLPath.__new__(_SCLPath)
        child.L = self.L
        child.B = self.B
        child.pm = self.pm
        child.parent_L = self
        child.parent_B = self
        child.copy_L = False
        child.copy_B = False
        return child


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            path.ensure_llr_copy()
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_llr = path.L[j - branch_size, s]
                    btm_llr = path.L[j, s]
                    top_bit = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = _lower_llr(top_llr, btm_llr, top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            path.ensure_bit_copy()
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if hard == bit else abs(llr_val)

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列（最优路径）
            pm: 最优路径的度量值
        """
        llr_internal = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = [_SCLPath(self.N, self.n, llr_internal)]

        for i in range(self.N):
            l = int(self.br[i])
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    penalty = self._path_metric_penalty(llr_val, 0)
                    new_path = path.fork()
                    new_path.ensure_bit_copy()
                    new_path.pm += penalty
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.fork()
                        new_path.ensure_bit_copy()
                        new_path.pm += self._path_metric_penalty(llr_val, bit)
                        new_path.B[l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = paths[0]
        u_hat = best.B[:, self.n].astype(int)

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            payload = u_hat[info_idx]
            valid = [p for p in paths if crc_check(p.B[:, self.n][info_idx], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)
                u_hat = best.B[:, self.n].astype(int)

        return u_hat, best.pm
