"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    bit_reversed,
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _frozen_index_set,
)


CRC_POLYNOMIALS = {
    8: [1, 0, 0, 0, 0, 1, 1, 1, 1],
    16: [1] + [(0x8005 >> i) & 1 for i in range(15, -1, -1)],
}


def _poly_crc_remainder(bits, generator):
    bits = [int(b) for b in bits]
    r = len(generator) - 1
    reg = bits + [0] * r
    for i in range(len(bits)):
        if reg[i] == 1:
            for j in range(len(generator)):
                reg[i + j] ^= generator[j]
    return reg[len(bits):]


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    generator = CRC_POLYNOMIALS[crc_length]
    remainder = _poly_crc_remainder(info_bits, generator)
    return np.concatenate([info_bits, np.array(remainder, dtype=int)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    generator = CRC_POLYNOMIALS[crc_length]
    remainder = _poly_crc_remainder(bits, generator)
    return all(v == 0 for v in remainder)


class _Path:
    __slots__ = ("pm", "L", "B", "parent", "fork_stage")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.parent = None
        self.fork_stage = -1


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _frozen_index_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(np.asarray(frozen_bits, dtype=int) == 0)[0]

    def _clone(self, path, copy_state=False):
        new_path = _Path(self.N, self.n)
        new_path.pm = path.pm
        if copy_state or path.parent is not None:
            new_path.L = path.L.copy()
            new_path.B = path.B.copy()
        else:
            new_path.L = path.L
            new_path.B = path.B
            new_path.parent = path
        return new_path

    def _ensure_owned(self, path, stage):
        if path.parent is not None and path.fork_stage <= stage:
            parent = path.parent
            path.L = parent.L.copy()
            path.B = parent.B.copy()
            path.parent = None
            path.fork_stage = -1

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            self._ensure_owned(path, s)
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            self._ensure_owned(path, s)
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen:
                    new_path = self._clone(path, copy_state=True)
                    new_path.pm += self._metric_penalty(llr, 0)
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone(path, copy_state=True)
                        new_path.pm += self._metric_penalty(llr, bit)
                        new_path.B[l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                payload = path.B[:, self.n].astype(int)[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm
