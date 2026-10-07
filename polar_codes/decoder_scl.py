"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed_int,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int64)
    rem = _crc_remainder(np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int64)]), crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int64
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int64)
    return _crc_remainder(bits, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr
        self.u_hat = np.zeros(N, dtype=np.int64)


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _advance_llr(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        top_bit,
                    )

    def _advance_bits(self, path, l):
        if l < self.N / 2:
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

    def _pm_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            from decoder_sc import sc_decode

            return sc_decode(llr_ch, self.frozen_bits), 0.0

        rev = bit_reversal_permutation(self.N)
        llr = llr_ch[rev]

        paths = [_Path(self.N, self.n, llr.copy())]

        for i in range(self.N):
            l = _bit_reversed_int(i, self.n)
            new_paths = []

            for path in paths:
                self._advance_llr(path, l)
                llr_leaf = path.L[l, self.n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(llr_leaf, 0)
                    path.pm += pen
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._advance_bits(path, l)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        p2 = _Path(self.N, self.n, path.L[:, 0].copy())
                        p2.L[:, 1:] = path.L[:, 1:].copy()
                        p2.B[:, 1:] = path.B[:, 1:].copy()
                        p2.pm = path.pm + self._pm_penalty(llr_leaf, u_bit)
                        p2.u_hat = path.u_hat.copy()
                        p2.u_hat[l] = u_bit
                        p2.B[l, self.n] = u_bit
                        self._advance_bits(p2, l)
                        new_paths.append(p2)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
