"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    prepare_channel_llr,
    _active_llr_level,
    _active_bit_level,
)
from encoder import bit_reversed


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length <= 8 else 1):
            if crc_length <= 8:
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
            else:
                if reg & (1 << (crc_length - 1)):
                    reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
                else:
                    reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
        rem = _crc_remainder(info_bits, poly, 8)
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    elif crc_length == 16:
        poly = _CRC16_POLY
        rem = _crc_remainder(info_bits, poly, 16)
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


# ==================== SCL 译码器 ====================

class _PathState:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n, llr):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        p = _PathState.__new__(_PathState)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        N, n = self.N, self.n
        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            branch = block // 2
            for j in range(l, N, block):
                if j % block < branch:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j, s], path.L[j - branch, s], path.B[j - branch, s + 1]
                    )

    def _update_bits(self, path, l):
        N, n = self.N, self.n
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block = 1 << s
            branch = block // 2
            for j in range(l, -1, -block):
                if j % block >= branch:
                    path.B[j - branch, s - 1] = path.B[j, s] ^ path.B[j - branch, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr = prepare_channel_llr(llr_ch)
        N, n, L = self.N, self.n, self.list_size
        paths = [_PathState(N, n, llr)]

        for phi_nat in range(N):
            l = bit_reversed(phi_nat, n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_phi = float(path.L[l, n])

                if self.frozen_bits[l]:
                    child = path.copy()
                    child.pm += self._penalty(llr_phi, 0)
                    child.u_hat[l] = 0
                    child.B[l, n] = 0
                    self._update_bits(child, l)
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm += self._penalty(llr_phi, bit)
                        child.u_hat[l] = bit
                        child.B[l, n] = bit
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:L]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
