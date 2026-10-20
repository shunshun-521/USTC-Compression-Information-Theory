"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from encoder import bit_reversed
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _lower_llr,
    _upper_llr,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后（MSB-first）"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly, mask = _CRC8_POLY, 0xFF
    elif crc_length == 16:
        poly, mask = _CRC16_POLY, 0xFFFF
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    top = 1 << (crc_length - 1)
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 0:
        return True
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_path(self, src):
        p = _Path(self.N, self.n)
        p.L[:] = src.L
        p.B[:] = src.B
        p.pm = src.pm
        p.u_hat[:] = src.u_hat
        return p

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phase in range(self.N):
            l = bit_reversed(phase, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    pen = 0.0 if llr >= 0 else abs(llr)
                    np_ = self._copy_path(path)
                    np_.pm += pen
                    np_.B[l, self.n] = 0
                    np_.u_hat[l] = 0
                    self._update_bits(np_, l)
                    new_paths.append(np_)
                else:
                    for bit in (0, 1):
                        np_ = self._copy_path(path)
                        if bit == 0:
                            pen = 0.0 if llr >= 0 else abs(llr)
                        else:
                            pen = 0.0 if llr < 0 else abs(llr)
                        np_.pm += pen
                        np_.B[l, self.n] = bit
                        np_.u_hat[l] = bit
                        self._update_bits(np_, l)
                        new_paths.append(np_)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        crc_ok = []
        for p in paths:
            if self.crc_length > 0:
                if crc_check(p.u_hat[self.info_indices], self.crc_length):
                    crc_ok.append(p)
            else:
                crc_ok.append(p)

        best = min(crc_ok if crc_ok else paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
