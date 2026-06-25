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
    _frozen_indices,
    _lower_llr,
    _prepare_llr,
    _upper_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_bits(bits, poly, width):
    """CRC LFSR 余数（信息位后附加 width 个零等价）。"""
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << (width + 1)) - 1)
        if reg & (1 << width):
            reg ^= poly
    return reg & ((1 << width) - 1)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    remainder = _crc_bits(padded, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_bits(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(_frozen_indices(frozen_bits))
        self.frozen_bits = np.asarray(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(
            ~self.frozen_bits if self.frozen_bits.dtype == bool else self.frozen_bits == 0
        )[0]

    def _pm_update(self, pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            pm += abs(llr)
        return pm

    def _compute_root_llr(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
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

    def _propagate_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr = _prepare_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            is_frozen = l in self.frozen_set
            candidates = []

            for path in paths:
                llr_root = self._compute_root_llr(path, l)
                if is_frozen:
                    child = self._clone_path(path)
                    child.pm = self._pm_update(child.pm, llr_root, 0)
                    child.B[l, self.n] = 0
                    child.u_hat[l] = 0
                    self._propagate_bits(child, l)
                    candidates.append(child)
                else:
                    for u_bit in (0, 1):
                        child = self._clone_path(path)
                        child.pm = self._pm_update(child.pm, llr_root, u_bit)
                        child.B[l, self.n] = u_bit
                        child.u_hat[l] = u_bit
                        self._propagate_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm

    def _clone_path(self, path):
        child = _Path(self.N, self.n, path.L[:, 0])
        child.pm = path.pm
        child.L = path.L.copy()
        child.B = path.B.copy()
        child.u_hat = path.u_hat.copy()
        return child
