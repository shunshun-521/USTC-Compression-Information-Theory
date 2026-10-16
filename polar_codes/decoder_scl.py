"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    sc_decode,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    f_operation,
    g_operation,
    _frozen_indices,
)
from encoder import bit_reversal_permutation_fast


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
    for b in np.asarray(bits, dtype=np.int8):
        feedback = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(b)) & mask
        if feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位"""
    bits = np.asarray(info_bits, dtype=np.int8)
    padded = np.concatenate([bits, np.zeros(crc_length, dtype=np.int8)])
    reg = _crc_remainder(padded, crc_length)
    out = np.zeros(len(bits) + crc_length, dtype=np.int8)
    out[: len(bits)] = bits
    for i in range(crc_length):
        out[len(bits) + i] = (reg >> (crc_length - 1 - i)) & 1
    return out


def crc_check(bits, crc_length=8):
    return _crc_remainder(bits, crc_length) == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        p = _Path.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 LLR/比特树）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = _frozen_indices(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u_bit):
        v = 0 if llr >= 0 else 1
        return 0.0 if u_bit == v else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation_fast(self.N)
        llr_perm = llr_ch[br]

        paths = [_Path(self.N, self.n, llr_perm)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path.L[l, self.n]
                if l in self.frozen_set:
                    u0 = 0
                    new_p = path.copy()
                    new_p.pm += self._pm_penalty(llr_bit, 0)
                    new_p.B[l, self.n] = 0
                    new_p.u_hat[l] = 0
                    self._update_bits(new_p, l)
                    candidates.append(new_p)
                else:
                    for u_cand in (0, 1):
                        new_p = path.copy()
                        new_p.pm += self._pm_penalty(llr_bit, u_cand)
                        new_p.B[l, self.n] = u_cand
                        new_p.u_hat[l] = u_cand
                        self._update_bits(new_p, l)
                        candidates.append(new_p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = paths[0]

        return best.u_hat.astype(np.int8), best.pm
