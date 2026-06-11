"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    top = 1 << crc_length
    for bit in bits:
        reg <<= 1
        if bit:
            reg ^= top
        if reg & top:
            reg ^= poly
    return reg & (top - 1)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


# ==================== SCL 译码器 ====================


class _PathState:
    __slots__ = ("L", "B", "pm", "parent", "copied")

    def __init__(self, N, n, llr_ch, parent=None):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.parent = parent
        self.copied = parent is None
        if parent is None:
            self.L[:, 0] = llr_ch
        else:
            self.copied = False

    def materialize(self):
        if self.copied or self.parent is None:
            return
        self.L[:] = self.parent.L
        self.B[:] = self.parent.B
        self.copied = True


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _llr_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = int(self.br[i])
            candidates = []

            for path in paths:
                path.materialize()
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    bit = 0
                    new_path = _PathState(self.N, self.n, llr_ch, parent=path)
                    new_path.pm = path.pm + self._llr_penalty(llr, bit)
                    new_path.L[:] = path.L
                    new_path.B[:] = path.B
                    new_path.B[l, self.n] = bit
                    _update_bits(new_path.B, l, self.n)
                    new_path.copied = True
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = _PathState(self.N, self.n, llr_ch, parent=path)
                        new_path.pm = path.pm + self._llr_penalty(llr, bit)
                        new_path.L[:] = path.L
                        new_path.B[:] = path.B
                        new_path.B[l, self.n] = bit
                        _update_bits(new_path.B, l, self.n)
                        new_path.copied = True
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            for path in paths:
                info_part = path.B[:, self.n][self.info_indices]
                if crc_check(info_part, self.crc_length):
                    return path.B[:, self.n].astype(int).copy(), path.pm

        best = paths[0]
        return best.B[:, self.n].astype(int).copy(), best.pm
