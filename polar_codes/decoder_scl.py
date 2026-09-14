"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_scalar,
    _lower_llr,
    _upper_llr,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc = 0
    mask = (1 << crc_length) - 1
    high = 1 << (crc_length - 1)
    for bit in info_bits:
        crc ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if crc & high:
                crc = ((crc << 1) ^ poly) & mask
            else:
                crc = (crc << 1) & mask
    crc_bits = np.array([(crc >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class _PathState:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)


class SCLDecoder:
    """
    SCL 译码器。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_state(self, src):
        dst = _PathState(self.N, self.n)
        dst.pm = src.pm
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        return dst

    def _update_llrs(self, state, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    state.L[j, s + 1] = _upper_llr(state.L[j, s], state.L[j + branch_size, s])
                else:
                    state.L[j, s + 1] = _lower_llr(
                        state.L[j, s],
                        state.L[j - branch_size, s],
                        int(state.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, state, l, bit):
        state.B[l, self.n] = bit
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    state.B[j - branch_size, s - 1] = int(state.B[j, s]) ^ int(
                        state.B[j - branch_size, s]
                    )
                    state.B[j, s - 1] = state.B[j, s]

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]
        paths = [_PathState(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed_scalar(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    new_path = self._copy_state(path)
                    new_path.pm += penalty
                    self._update_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and llr_val >= 0) or (
                            bit == 1 and llr_val < 0
                        ) else abs(llr_val)
                        new_path = self._copy_state(path)
                        new_path.pm += penalty
                        self._update_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.B[:, self.n].astype(int)[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, self.n].astype(int), best.pm
