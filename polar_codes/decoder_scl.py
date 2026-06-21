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
    _channel_llr_to_decoder,
    _lower_llr,
    _upper_llr,
)
from encoder import bit_reversal_permutation


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005（MSB 先行）
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
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
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length <= 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


def _path_metric_update(pm, llr, bit):
    hard = 0 if llr >= 0 else 1
    penalty = 0.0 if bit == hard else abs(llr)
    return pm + penalty


class _Path:
    __slots__ = ("L", "B", "pm", "owner_id")

    def __init__(self, N, n, llr_in, owner_id):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_in
        self.pm = 0.0
        self.owner_id = owner_id


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits.astype(bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self._next_owner = 0

    def _new_path(self, llr_in):
        self._next_owner += 1
        return _Path(self.N, self.n, llr_in, self._next_owner)

    def _clone_path(self, path):
        new_path = _Path(self.N, self.n, path.L[:, 0].copy(), self._next_owner)
        self._next_owner += 1
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        return new_path

    def _ensure_owner(self, path, l, s, paths):
        if path.owner_id == paths[0].owner_id:
            return path
        return self._clone_path(path)

    def decode(self, llr_ch):
        llr_in = _channel_llr_to_decoder(llr_ch, self.N)
        paths = [self._new_path(llr_in)]

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            is_frozen = l in self.frozen_set
            candidates = []

            for path in paths:
                self._update_llrs(path, l)

                if is_frozen:
                    llr_val = path.L[l, self.n]
                    new_path = self._clone_path(path)
                    new_path.pm = _path_metric_update(path.pm, llr_val, 0)
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    llr_val = path.L[l, self.n]
                    for bit in (0, 1):
                        new_path = self._clone_path(path)
                        new_path.pm = _path_metric_update(path.pm, llr_val, bit)
                        new_path.B[l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_idx = np.where(self.frozen_bits == 0)[0]
            valid = []
            for path in paths:
                u = path.B[:, self.n].astype(int)
                payload = u[info_idx]
                if crc_check(payload, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 2 ** (s + 1)
            half = block // 2
            for j in range(l, self.N, block):
                if j % block < half:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + half, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - half, s],
                        int(path.B[j - half, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 2 ** s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    path.B[j - half, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - half, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]
