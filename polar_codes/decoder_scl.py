"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import copy
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_divide(data_bits, poly, crc_len):
    reg = [0] * crc_len
    for bit in data_bits:
        fb = bit ^ reg[0]
        reg = reg[1:] + [0]
        if fb:
            for i in range(crc_len):
                if (poly >> (crc_len - 1 - i)) & 1:
                    reg[i] ^= fb
    return np.array(reg, dtype=int)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_divide(info_bits, poly, crc_length)
    return np.concatenate([info_bits, rem])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 0:
        return True
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.array_equal(
        _crc_divide(bits[:-crc_length], poly, crc_length), bits[-crc_length:]
    )


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_update(pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            return pm + abs(llr)
        return pm

    def _update_llrs(self, L, B, l):
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            half = block // 2
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - half, s], L[j, s], B[j - half, s + 1]
                    )

    def _update_bits(self, B, l):
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block = 1 << s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    B[j - half, s - 1] = B[j, s] ^ B[j - half, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, Lsz = self.N, self.n, self.L_size

        paths = [_Path(N, n, llr_ch)]

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []

            for path in paths:
                self._update_llrs(path.L, path.B, l)
                cur_llr = path.L[l, n]

                if self.frozen_bits[l]:
                    child = copy.deepcopy(path)
                    child.pm = self._pm_update(path.pm, cur_llr, 0)
                    child.B[l, n] = 0
                    child.u_hat[l] = 0
                    self._update_bits(child.B, l)
                    new_paths.append(child)
                else:
                    for u_val in (0, 1):
                        child = copy.deepcopy(path)
                        child.pm = self._pm_update(path.pm, cur_llr, u_val)
                        child.B[l, n] = u_val
                        child.u_hat[l] = u_val
                        self._update_bits(child.B, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:Lsz]

        crc_pass = []
        for idx, p in enumerate(paths):
            if self.crc_length > 0:
                bits = p.u_hat[self.info_indices]
                if crc_check(bits, self.crc_length):
                    crc_pass.append(idx)

        if self.crc_length > 0 and crc_pass:
            best = min(crc_pass, key=lambda idx: paths[idx].pm)
        else:
            best = 0

        return paths[best].u_hat, paths[best].pm
