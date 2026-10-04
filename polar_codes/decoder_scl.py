"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from encoder import bit_reversed
from decoder_sc import (
    f_operation,
    g_operation,
    _frozen_mask,
    _active_llr_level,
    _active_bit_level,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(info_bits, poly, crc_length):
    reg = 0
    for bit in info_bits:
        reg = ((reg << 1) | int(bit)) & ((1 << (crc_length + 1)) - 1)
        if reg & (1 << crc_length):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & ((1 << (crc_length + 1)) - 1)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=int)


def _path_update_llrs(path, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
            else:
                top_bit = path.B[j - branch_size, s + 1]
                path.L[j, s + 1] = g_operation(
                    path.L[j - branch_size, s], path.L[j, s], top_bit
                )


def _path_update_bits(path, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.B[j - branch_size, s - 1] = (
                    path.B[j, s] ^ path.B[j - branch_size, s]
                )
                path.B[j, s - 1] = path.B[j, s]


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _frozen_mask(frozen_bits)
        self.frozen_set = set(np.where(self.frozen)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                _path_update_llrs(path, l, self.n, self.N)
                llr = path.L[l, self.n]
                hard = 0 if llr >= 0 else 1

                if l in self.frozen_set:
                    pen = 0.0 if hard == 0 else abs(llr)
                    p = self._clone(path)
                    p.pm += pen
                    p.B[l, self.n] = 0
                    p.u_hat[l] = 0
                    _path_update_bits(p, l, self.n, self.N)
                    candidates.append(p)
                else:
                    for u in (0, 1):
                        p = self._clone(path)
                        pen = 0.0 if u == hard else abs(llr)
                        p.pm += pen
                        p.B[l, self.n] = u
                        p.u_hat[l] = u
                        _path_update_bits(p, l, self.n, self.N)
                        candidates.append(p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = None
        best_pm = None
        for p in paths:
            if self.crc_length > 0:
                if not crc_check(p.u_hat[self.info_indices], self.crc_length):
                    continue
            if best_pm is None or p.pm < best_pm:
                best_pm = p.pm
                best = p

        if best is None:
            best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm

    @staticmethod
    def _clone(path):
        p = _Path.__new__(_Path)
        p.pm = path.pm
        p.L = path.L.copy()
        p.B = path.B.copy()
        p.u_hat = path.u_hat.copy()
        return p
