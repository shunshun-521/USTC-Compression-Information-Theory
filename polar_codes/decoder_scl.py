"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _lower_llr,
    _reorder_channel_llr,
    _upper_llr,
    precompute_sc_indices,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top_bit = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & top_bit:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    expected = bits[-crc_length:]
    remainder = _crc_remainder(payload, poly, crc_length)
    actual = sum(int(bit) << (crc_length - 1 - i) for i, bit in enumerate(expected))
    return remainder == actual


class Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat', 'active')

    def __init__(self, n, N):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        _, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _update_llr(self, path, leaf):
        for stage in range(self.n - _active_llr_level(leaf, self.n), self.n):
            block_size = 1 << (stage + 1)
            branch_size = block_size >> 1
            for j in range(leaf, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, stage + 1] = _upper_llr(path.L[j, stage], path.L[j + branch_size, stage])
                else:
                    path.L[j, stage + 1] = _lower_llr(
                        path.L[j, stage],
                        path.L[j - branch_size, stage],
                        path.B[j - branch_size, stage + 1],
                    )

    def _propagate_bits(self, path, leaf):
        if leaf < self.N // 2:
            return
        for stage in range(self.n, self.n - _active_bit_level(leaf, self.n), -1):
            block_size = 1 << stage
            branch_size = block_size >> 1
            for j in range(leaf, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, stage - 1] = path.B[j, stage] ^ path.B[j - branch_size, stage]
                    path.B[j, stage - 1] = path.B[j, stage]

    def decode(self, llr_ch):
        llr_tree = _reorder_channel_llr(llr_ch, self.N)
        paths = [Path(self.n, self.N)]
        paths[0].L[:, 0] = llr_tree

        for phi in range(self.N):
            leaf = _bit_reversed_index(phi, self.n)
            candidates = []

            for path in paths:
                if not path.active:
                    continue

                self._update_llr(path, leaf)
                llr = path.L[leaf, self.n]

                if leaf in self.frozen_set:
                    path.pm += self._pm_penalty(llr, 0)
                    path.u_hat[leaf] = 0
                    path.B[leaf, self.n] = 0
                    self._propagate_bits(path, leaf)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        new_path = Path(self.n, self.N)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.pm = path.pm + self._pm_penalty(llr, bit)
                        new_path.u_hat = path.u_hat.copy()
                        new_path.u_hat[leaf] = bit
                        new_path.B[leaf, self.n] = bit
                        self._propagate_bits(new_path, leaf)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = paths[0]

        return best.u_hat.astype(int), best.pm
