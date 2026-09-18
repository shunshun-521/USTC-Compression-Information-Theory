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
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


class _PathState:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new = _PathState.__new__(_PathState)
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
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
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = path.copy()
                    new_path.pm += abs(llr_val) if llr_val < 0 else 0.0
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        if (bit == 0 and llr_val < 0) or (bit == 1 and llr_val >= 0):
                            new_path.pm += abs(llr_val)
                        new_path.B[l, self.n] = bit
                        new_path.u_hat[l] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_pass = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(crc_pass, key=lambda p: p.pm) if crc_pass else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
