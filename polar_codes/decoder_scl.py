"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    lower_llr,
    upper_llr,
)
from encoder import bit_reversed_index, prepare_decoder_llr


CRC_POLYS = {8: 0x07, 16: 0x8005}


def _bits_to_int(bits):
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    val = _bits_to_int(info_bits)
    n = len(info_bits)
    val_shifted = val << crc_length
    poly_full = (1 << crc_length) | poly
    for i in range(n + crc_length - 1, crc_length - 1, -1):
        if (val_shifted >> i) & 1:
            val_shifted ^= poly_full << (i - crc_length)
    crc_val = val_shifted & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(crc_val >> (crc_length - 1 - j)) & 1 for j in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    val = _bits_to_int(bits)
    n = len(bits)
    poly_full = (1 << crc_length) | poly
    for i in range(n - 1, crc_length - 1, -1):
        if (val >> i) & 1:
            val ^= poly_full << (i - crc_length)
    return (val & ((1 << crc_length) - 1)) == 0


class Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        p = Path.__new__(Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = prepare_decoder_llr(llr_ch)
        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen:
                    new_path = path.copy()
                    if llr < 0:
                        new_path.pm += abs(llr)
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = path.copy()
                        if (u_bit == 0 and llr < 0) or (u_bit == 1 and llr >= 0):
                            new_path.pm += abs(llr)
                        new_path.u_hat[l] = u_bit
                        new_path.B[l, self.n] = u_bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_all = paths[0]
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path.pm < best_crc.pm:
                        best_crc = path

        chosen = best_crc if best_crc is not None else best_all
        return chosen.u_hat.astype(int), chosen.pm
