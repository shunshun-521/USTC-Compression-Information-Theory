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
    _frozen_set_from_mask,
    f_operation,
    g_operation,
)
from encoder import prepare_channel_llr

# ==================== CRC 工具 ====================

CRC_POLYS = {
    8: 0x107,   # x^8 + x^2 + x + 1
    16: 0x11021,  # CRC-16-IBM
}


def _crc_remainder(bits, poly, crc_length):
    """多项式长除法求 CRC 余数。"""
    msg = np.concatenate([bits, np.zeros(crc_length, dtype=np.int8)])
    for i in range(len(bits)):
        if msg[i]:
            for shift in range(crc_length, -1, -1):
                if poly & (1 << shift):
                    msg[i + shift] ^= 1
    return msg[-crc_length:]


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    crc_bits = _crc_remainder(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    poly = CRC_POLYS[crc_length]
    remainder = _crc_remainder(bits[:-crc_length], poly, crc_length)
    return np.array_equal(bits[-crc_length:], remainder)


# ==================== SCL 译码器 ====================


class _SCLPath:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        other = _SCLPath(self.L.shape[0], int(math.log2(self.L.shape[0])))
        other.L = self.L.copy()
        other.B = self.B.copy()
        other.pm = self.pm
        other.u_hat = self.u_hat.copy()
        return other


class SCLDecoder:
    """
    SCL 译码器（Permuted SCD 扩展）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = _frozen_set_from_mask(self.frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is None:
            self.info_indices = np.where(~self.frozen_bits)[0]
        else:
            self.info_indices = np.asarray(info_indices, dtype=int)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_llr = path.L[j - branch_size, s]
                    btm_llr = path.L[j, s]
                    top_bit = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr = prepare_channel_llr(llr_ch)
        paths = [_SCLPath(self.N, self.n)]
        paths[0].L[:, 0] = llr

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm += self._penalty(llr_val, 0)
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm += self._penalty(llr_val, bit)
                        child.B[l, self.n] = bit
                        child.u_hat[l] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
