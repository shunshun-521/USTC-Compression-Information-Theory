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
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


def _crc_remainder(bits, crc_length=8):
    """计算 CRC 余数（标准移位寄存器，一位一移）"""
    bits = np.asarray(bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in np.concatenate([bits, np.zeros(crc_length, dtype=int)]):
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & mask) ^ (fb * poly)
    return reg


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否为 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.pm = 0.0


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _branch_pm(self, pm, llr, bit):
        if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0):
            return pm
        return pm + abs(llr)

    def _init_paths(self, llr_ch):
        br = bit_reversal_permutation(self.N)
        path = _Path(self.N, self.n)
        path.L[:, 0] = llr_ch[br]
        return [path]

    def _extend_paths(self, paths, l):
        n = self.n
        candidates = []
        for path in paths:
            _update_llrs(path.L, path.B, l, n)
            llr = path.L[l, n]
            if self.frozen_bits[l]:
                child = _Path(self.N, n)
                child.L[:] = path.L
                child.B[:] = path.B
                child.pm = self._branch_pm(path.pm, llr, 0)
                child.B[l, n] = 0
                _update_bits(child.B, l, n)
                candidates.append(child)
            else:
                for bit in (0, 1):
                    child = _Path(self.N, n)
                    child.L[:] = path.L
                    child.B[:] = path.B
                    child.pm = self._branch_pm(path.pm, llr, bit)
                    child.B[l, n] = bit
                    _update_bits(child.B, l, n)
                    candidates.append(child)
        candidates.sort(key=lambda p: p.pm)
        return candidates[: self.list_size]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._init_paths(llr_ch)

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            paths = self._extend_paths(paths, l)

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.B[:, self.n][self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm
