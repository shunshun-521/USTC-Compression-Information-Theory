"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    f_boxplus,
    g_operation,
    precompute_sc_indices,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly

    crc_bits = np.zeros(crc_length, dtype=int)
    for i in range(crc_length - 1, -1, -1):
        reg <<= 1
        if reg & (1 << crc_length):
            reg ^= poly
            crc_bits[i] = 1

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return (reg & ((1 << crc_length) - 1)) == 0


class Path:
    """SCL 单条路径。"""

    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, L, B, l):
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        L_size = self.list_size

        paths = [Path(N, n)]
        paths[0].L[:, 0] = llr_ch.copy()

        for phi in range(N):
            l = _bit_reversed(phi, n)
            candidates = []

            for path in paths:
                self._update_llrs(path.L, path.B, l)
                llr = path.L[l, n]

                if self.frozen_bits[l]:
                    new_path = self._lazy_copy(path)
                    new_path.pm += self._path_metric_penalty(llr, 0)
                    new_path.B[l, n] = 0
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._lazy_copy(path)
                        new_path.pm += self._path_metric_penalty(llr, bit)
                        new_path.B[l, n] = bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:L_size]

            for path in paths:
                self._update_bits(path.B, l)

        best_path = self._select_best_path(paths)
        return best_path.B[:, n].astype(int).copy(), best_path.pm

    def _lazy_copy(self, path):
        new_path = Path(self.N, self.n)
        new_path.pm = path.pm
        new_path.L = path.L
        new_path.B = path.B.copy()
        return new_path

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.B[:, self.n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                return min(valid, key=lambda p: p.pm)
        return min(paths, key=lambda p: p.pm)
