"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Permuted SC
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _lower_llr,
    _update_bits,
    _update_llrs,
    f_operation,
)
from encoder import bit_reversed


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
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
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch

    def copy(self):
        p = Path.__new__(Path)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Permuted SC + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        paths = [Path(N, n, llr_ch)]

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n)
                llr_bit = path.L[l, n]

                if self.frozen_bits[l]:
                    u = 0
                    penalty = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    new_path = path.copy()
                    new_path.pm += penalty
                    new_path.B[l, n] = 0
                    _update_bits(new_path.B, l, n, N)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        expected = 0 if llr_bit >= 0 else 1
                        penalty = 0.0 if u == expected else abs(llr_bit)
                        new_path = path.copy()
                        new_path.pm += penalty
                        new_path.B[l, n] = u
                        _update_bits(new_path.B, l, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.B[:, n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, n].astype(int), best.pm
