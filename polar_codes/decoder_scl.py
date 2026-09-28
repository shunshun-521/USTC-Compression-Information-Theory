"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _SCDCore,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([np.asarray(info_bits, dtype=int), np.asarray(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    if crc_length == 0:
        return True
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


class SCLDecoder:
    """SCL 译码器（基于 SCD 因子图，Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is None:
            self.info_indices = np.where(~self.frozen_bits)[0]
        else:
            self.info_indices = np.asarray(info_indices, dtype=np.int64)
        self.br = bit_reversal_permutation(N)

    def _path_llr(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )
        return path.L[l, self.n]

    def _path_update_bits(self, path, l):
        if l < self.N / 2:
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

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]

        class Path:
            __slots__ = ("L", "B", "pm", "copied_L", "copied_B")

            def __init__(self, llr):
                self.L = np.full((self_N, self_n + 1), np.nan, dtype=np.float64)
                self.B = np.full((self_N, self_n + 1), np.nan)
                self.L[:, 0] = llr
                self.pm = 0.0
                self.copied_L = False
                self.copied_B = False

        self_N, self_n = self.N, self.n
        paths = [Path(llr_ch)]

        decode_order = [_bit_reversed(i, self.n) for i in range(self.N)]

        for l in decode_order:
            expanded = []
            for path in paths:
                if not path.copied_L:
                    path.L = path.L.copy()
                    path.copied_L = True
                llr = self._path_llr(path, l)

                if l in self.frozen_set:
                    pen = self._pm_penalty(llr, 0)
                    if not path.copied_B:
                        path.B = path.B.copy()
                        path.copied_B = True
                    path.B[l, self.n] = 0
                    path.pm += pen
                    self._path_update_bits(path, l)
                    expanded.append(path)
                else:
                    for u in (0, 1):
                        child = Path.__new__(Path)
                        child.L = path.L
                        child.B = path.B.copy()
                        child.pm = path.pm + self._pm_penalty(llr, u)
                        child.copied_L = True
                        child.copied_B = True
                        child.B[l, self.n] = u
                        self._path_update_bits(child, l)
                        expanded.append(child)

            expanded.sort(key=lambda p: p.pm)
            paths = expanded[: self.list_size]

        best_crc = None
        best_pm = None
        for path in paths:
            u_hat = path.B[:, self.n].astype(int)
            if self.crc_length > 0:
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    if best_crc is None or path.pm < best_crc.pm:
                        best_crc = path
            if best_pm is None or path.pm < best_pm.pm:
                best_pm = path

        chosen = best_crc if best_crc is not None else best_pm
        return chosen.B[:, self.n].astype(int), chosen.pm
