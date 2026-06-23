"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _channel_llr_layout,
    f_operation,
    g_operation,
    sc_decode,
)
from encoder import bit_reversal_permutation

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_divide(info_bits, poly, crc_length):
    bits = np.asarray(info_bits, dtype=np.int8).tolist()
    reg = [0] * crc_length
    poly_bits = [(poly >> i) & 1 for i in range(crc_length, -1, -1)][1:]

    for bit in bits:
        msb = reg.pop(0)
        reg.append(bit ^ msb)
        if msb:
            reg = [reg[i] ^ poly_bits[i] for i in range(crc_length)]

    return np.array(reg, dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_divide(info_bits, poly, crc_length)
    return np.concatenate([np.asarray(info_bits, dtype=np.int8), remainder])


def crc_check(bits, crc_length=8):
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_divide(bits[:-crc_length], poly, crc_length)
    return np.array_equal(remainder, bits[-crc_length:])


# ==================== SCL 译码器 ====================


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 1 << (s + 1)
            half = block >> 1
            for j in range(l, self.N, block):
                if j % block < half:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + half, s])
                else:
                    top_bit = path.B[j - half, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - half, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l, bit):
        path.B[l, self.n] = bit
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 1 << s
            half = block >> 1
            for j in range(l, -1, -block):
                if j % block >= half:
                    path.B[j - half, s - 1] = path.B[j, s] ^ path.B[j - half, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_layout = _channel_llr_layout(llr_ch, self.N)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_layout

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    path.pm += self._pm_penalty(llr, 0)
                    self._update_bits(path, l, 0)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N, self.n)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = path.pm + self._pm_penalty(llr, bit)
                        self._update_bits(child, l, bit)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = path.B[:, self.n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            chosen = (
                min(valid, key=lambda p: p.pm)
                if valid
                else min(paths, key=lambda p: p.pm)
            )
        else:
            chosen = min(paths, key=lambda p: p.pm)

        u_hat = chosen.B[:, self.n].astype(int)
        return u_hat, chosen.pm
