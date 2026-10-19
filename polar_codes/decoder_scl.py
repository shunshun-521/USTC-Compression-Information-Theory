"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    sc_decode,
    sc_path_metric_penalty,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
)
from encoder import _bit_reversal_permutation_fast

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        feedback = ((reg >> (crc_length - 1)) & 1) ^ int(b)
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _PathState:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = _bit_reversal_permutation_fast(N)
        self.phases = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _init_paths(self, llr_ch):
        p = _PathState(self.N, self.n)
        for j in range(self.N):
            p.L[j, 0] = llr_ch[self.br[j]]
        return [p]

    def _update_llrs(self, path, l):
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
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._init_paths(llr_ch)

        for l in self.phases:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr_leaf = path.L[l, self.n]
                if l in self.frozen_set:
                    pen = sc_path_metric_penalty(llr_leaf, 0)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for u_bit in (0, 1):
                        p2 = _PathState(self.N, self.n)
                        p2.L = path.L.copy()
                        p2.B = path.B.copy()
                        p2.pm = path.pm + sc_path_metric_penalty(llr_leaf, u_bit)
                        p2.B[l, self.n] = u_bit
                        self._update_bits(p2, l)
                        candidates.append(p2)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        u_hat = paths[0].B[:, self.n].astype(np.int8)
        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.B[:, self.n][self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)
                u_hat = best.B[:, self.n].astype(np.int8)
                return u_hat, best.pm
        return u_hat, paths[0].pm
