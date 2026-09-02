"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed_index,
)


# CRC-8 (0x07), CRC-16 (0x8005)
_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_encode_bits(info_bits, poly, crc_length):
    reg = [0] * crc_length
    for bit in info_bits:
        fb = int(bit) ^ reg[0]
        reg = reg[1:] + [0]
        if fb:
            poly_bits = [(poly >> i) & 1 for i in range(crc_length - 1, -1, -1)]
            reg = [r ^ p for r, p in zip(reg, poly_bits)]
    return np.array(reg, dtype=int)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        crc_bits = _crc_encode_bits(info_bits, _CRC8_POLY, 8)
    elif crc_length == 16:
        crc_bits = _crc_encode_bits(info_bits, _CRC16_POLY, 16)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = [0] * crc_length
    for bit in bits:
        fb = int(bit) ^ reg[0]
        reg = reg[1:] + [0]
        if fb:
            poly_bits = [(poly >> i) & 1 for i in range(crc_length - 1, -1, -1)]
            reg = [r ^ p for r, p in zip(reg, poly_bits)]
    return sum(reg) == 0


class _Path:
  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n):
    self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=np.int_)
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=np.int_)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.brp = bit_reversal_permutation(N)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
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
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _current_llr(self, path, l):
        return path.L[l, self.n]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.brp]
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = self._current_llr(path, l)

                if self.frozen_bits[l]:
                    pen = self._pm_penalty(llr, 0)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N, self.n)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.u_hat = path.u_hat.copy()
                        child.pm = path.pm + self._pm_penalty(llr, bit)
                        child.B[l, self.n] = bit
                        child.u_hat[l] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_pass(p)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm

    def _crc_pass(self, path):
        info_bits = path.u_hat[~self.frozen_bits]
        if len(info_bits) < self.crc_length:
            return False
        return crc_check(info_bits, self.crc_length)
