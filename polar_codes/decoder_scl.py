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
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError(f'Unsupported CRC length: {crc_length}')


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)]
    return np.concatenate([info_bits.astype(int), np.array(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg == 0


def _natural_to_tree_frozen(frozen_bits):
    br = bit_reversal_permutation(len(frozen_bits))
    return frozen_bits[br].astype(bool)


def _pm_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    return pm if u == u_hard else pm + abs(llr)


class _Path:
    __slots__ = ('pm', 'L', 'B', 'decoded')

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.decoded = {}


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_tree = _natural_to_tree_frozen(frozen_bits)
        self.frozen_bits = frozen_bits.astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.info_natural = np.where(~self.frozen_bits)[0]

    def _copy_path(self, src, dst):
        dst.pm = src.pm
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.decoded = src.decoded.copy()

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(path.L[j - branch_size, s], path.L[j, s], top_bit)

    def _update_bits(self, path, l, u_bit):
        path.B[l, self.n] = u_bit
        path.decoded[l] = u_bit
        if l >= self.N // 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                        path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch
        pool = [_Path(self.N, self.n) for _ in range(self.list_size)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if self.frozen_tree[l]:
                    candidates.append((_pm_update(path.pm, llr, 0), path, 0))
                else:
                    for u in (0, 1):
                        candidates.append((_pm_update(path.pm, llr, u), path, u))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[:self.list_size]

            new_paths = []
            for idx, (pm, parent, u_bit) in enumerate(selected):
                child = pool[idx]
                self._copy_path(parent, child)
                child.pm = pm
                self._update_bits(child, l, u_bit)
                new_paths.append(child)
            paths = new_paths

        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            crc_paths = []
            for p in paths:
                u_nat = np.zeros(self.N, dtype=int)
                for tree_idx, bit in p.decoded.items():
                    u_nat[self.br[tree_idx]] = bit
                if crc_check(u_nat[self.info_natural], self.crc_length):
                    crc_paths.append(p)
            if crc_paths:
                best = min(crc_paths, key=lambda p: p.pm)

        u_nat = np.zeros(self.N, dtype=int)
        for tree_idx, bit in best.decoded.items():
            u_nat[self.br[tree_idx]] = bit
        return u_nat, best.pm
