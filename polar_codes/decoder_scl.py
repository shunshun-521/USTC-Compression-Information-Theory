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
    f_operation,
    g_operation,
    precompute_sc_indices,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_shift(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_shift(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_shift(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ('L', 'B', 'pm')

    def __init__(self, n, N):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.brp = bit_reversal_permutation(N)
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        _, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def _update_llrs(self, path, i):
        n = self.n
        N = self.N
        L = path.L
        B = path.B
        l = _bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _propagate_bits(self, path, i):
        n = self.n
        N = self.N
        B = path.B
        l = _bit_reversed(i, n)
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        active = [_Path(n, N)]
        active[0].L[:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in active:
                self._update_llrs(path, i)
                llr = path.L[l, n]

                if self.frozen_bits[i]:
                    new_path = _Path(n, N)
                    new_path.L[:] = path.L
                    new_path.B[:] = path.B
                    new_path.pm = path.pm + self._path_metric_penalty(llr, 0)
                    new_path.B[l, n] = 0
                    self._propagate_bits(new_path, i)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = _Path(n, N)
                        new_path.L[:] = path.L
                        new_path.B[:] = path.B
                        new_path.pm = path.pm + self._path_metric_penalty(llr, bit)
                        new_path.B[l, n] = bit
                        self._propagate_bits(new_path, i)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            active = candidates[: self.list_size]

        best = active[0]
        if self.crc_length > 0:
            crc_ok = []
            for p in active:
                u_hat = p.B[:, n].astype(int)[self.brp]
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    crc_ok.append(p)
            if crc_ok:
                best = min(crc_ok, key=lambda p: p.pm)

        u_hat = best.B[:, n].astype(int)[self.brp]
        return u_hat, best.pm
