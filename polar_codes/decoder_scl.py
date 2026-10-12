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
    _lower_llr,
    _update_bits,
    _update_llrs,
    _upper_llr,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class PathState:
    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_llr_at(self, path, phi):
        l = _bit_reversed(phi, self.n)
        _update_llrs(path.L, path.B, l, self.n)
        return path.L[l, self.n]

    def _path_propagate_bit(self, path, phi, u_bit):
        l = _bit_reversed(phi, self.n)
        path.B[l, self.n] = u_bit
        _update_bits(path.B, l, self.n)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [PathState(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []
            for path in paths:
                if l in self.frozen_set:
                    work = self._clone(path)
                    llr0 = self._path_llr_at(work, phi)
                    penalty = 0.0 if llr0 >= 0 else abs(llr0)
                    work.pm += penalty
                    self._path_propagate_bit(work, phi, 0)
                    candidates.append(work)
                else:
                    for u_bit in (0, 1):
                        work = self._clone(path)
                        llr0 = self._path_llr_at(work, phi)
                        consistent = (u_bit == 0 and llr0 >= 0) or (u_bit == 1 and llr0 < 0)
                        penalty = 0.0 if consistent else abs(llr0)
                        work.pm += penalty
                        self._path_propagate_bit(work, phi, u_bit)
                        candidates.append(work)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = self._select_best_path(paths)
        u_hat = best.B[:, self.n].astype(int)
        return u_hat, best.pm

    def _clone(self, path):
        new_p = PathState(self.N, self.n, path.L[:, 0])
        new_p.pm = path.pm
        new_p.u_hat = path.u_hat.copy()
        new_p.L = path.L.copy()
        new_p.B = path.B.copy()
        return new_p

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.B[:, self.n].astype(int)[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                return min(valid, key=lambda p: p.pm)
        return min(paths, key=lambda p: p.pm)
