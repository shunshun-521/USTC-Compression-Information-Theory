"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    bit_reversed_int,
    _frozen_mask,
    _update_llr,
    _update_bits,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, n] = llr


class SCLDecoder:
    """SCL 译码器（路径分裂时 deepcopy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = _frozen_mask(frozen_bits)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    @staticmethod
    def _pm_update(pm, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        if u_bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]
        paths = [_Path(self.N, self.n, llr)]

        for i in range(self.N):
            l = bit_reversed_int(i, self.n)
            new_paths = []
            for path in paths:
                _update_llr(path.L, path.B, l, self.n)
                llr_bit = path.L[l, 0]
                if l in self.frozen_set:
                    p = copy.deepcopy(path)
                    p.pm = self._pm_update(p.pm, llr_bit, 0)
                    p.B[l, 0] = 0
                    _update_bits(p.B, l, self.n)
                    new_paths.append(p)
                else:
                    for u_bit in (0, 1):
                        p = copy.deepcopy(path)
                        p.pm = self._pm_update(p.pm, llr_bit, u_bit)
                        p.B[l, 0] = u_bit
                        _update_bits(p.B, l, self.n)
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = p.B[:, 0][self.info_indices]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, 0].astype(int), best.pm
