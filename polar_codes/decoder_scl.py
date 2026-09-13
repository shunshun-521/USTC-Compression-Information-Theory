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
    _lower_llr,
    _upper_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_compute(info_bits, crc_length=8):
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in info_bits:
        crc ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 16):
            if crc & top:
                crc = ((crc << 1) ^ poly) & mask
            else:
                crc = (crc << 1) & mask
    return crc


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    crc_val = _crc_compute(info_bits, crc_length)
    crc_bits = np.array(
        [(crc_val >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    data = bits[:-crc_length]
    received = bits[-crc_length:]
    crc_val = _crc_compute(data, crc_length)
    expected = np.array(
        [(crc_val >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.array_equal(received, expected)


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch


class SCLDecoder:
    """SCL 译码器（Permuted SCD + 路径裁剪）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L = self.N, self.n, self.list_size

        paths = [_Path(N, n, llr_ch)]

        for phi in range(N):
            l = _bit_reversed(phi, n)
            candidates = []

            for p_idx, path in enumerate(paths):
                self._update_llrs(path, l)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    penalty = self._pm_penalty(llr, 0)
                    path.pm += penalty
                    path.B[l, n] = 0
                    candidates.append((path.pm, p_idx, None))
                else:
                    for bit in (0, 1):
                        penalty = self._pm_penalty(llr, bit)
                        candidates.append((path.pm + penalty, p_idx, bit))

            candidates.sort(key=lambda x: x[0])
            new_paths = []

            for pm, src_idx, bit in candidates[:L]:
                if bit is None:
                    path = paths[src_idx]
                    path.pm = pm
                else:
                    path = _Path(N, n, llr_ch)
                    src = paths[src_idx]
                    path.pm = pm
                    path.L[:] = src.L
                    path.B[:] = src.B
                    path.B[l, n] = bit
                new_paths.append(path)

            paths = new_paths

            for path in paths:
                self._update_bits(path, l)

        if self.crc_length > 0:
            valid = []
            for i, path in enumerate(paths):
                info_bits = path.B[:, n].astype(int)[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(i)
            best = min(valid, key=lambda i: paths[i].pm) if valid else min(
                range(len(paths)), key=lambda i: paths[i].pm
            )
        else:
            best = min(range(len(paths)), key=lambda i: paths[i].pm)

        u_hat = paths[best].B[:, n].astype(int)
        return u_hat, paths[best].pm
