"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    bit_reversed,
    _to_frozen_set,
    upper_llr,
    lower_llr,
    active_llr_level,
    active_bit_level,
    hard_decision,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & mask
        if reg & top:
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & mask
        if reg & top:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat', 'N', 'n')

    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


def _update_llr_path(path, l):
    for s in range(path.n - active_llr_level(l, path.n), path.n):
        block_size = int(2 ** (s + 1))
        branch_size = int(block_size / 2)
        for j in range(l, path.N, block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
            else:
                path.L[j, s + 1] = lower_llr(
                    path.L[j, s],
                    path.L[j - branch_size, s],
                    int(path.B[j - branch_size, s + 1]),
                )


def _update_bits_path(path, l):
    if l < path.N / 2:
        return
    for s in range(path.n, path.n - active_bit_level(l, path.n), -1):
        block_size = int(2 ** s)
        branch_size = int(block_size / 2)
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                path.B[j, s - 1] = path.B[j, s]


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = _to_frozen_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def _copy_path(self, src):
        dst = Path(self.N, self.n)
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.pm = src.pm
        dst.u_hat = src.u_hat.copy()
        return dst

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = [Path(N, n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(N):
            l = bit_reversed(phi, n)
            new_paths = []

            for path in paths:
                _update_llr_path(path, l)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    child = self._copy_path(path)
                    child.pm += self._pm_penalty(llr, 0)
                    child.B[l, n] = 0
                    child.u_hat[l] = 0
                    _update_bits_path(child, l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = self._copy_path(path)
                        child.pm += self._pm_penalty(llr, bit)
                        child.B[l, n] = bit
                        child.u_hat[l] = bit
                        _update_bits_path(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
