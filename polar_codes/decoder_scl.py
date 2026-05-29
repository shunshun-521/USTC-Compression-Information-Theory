"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np
from encoder import bit_reversed_index
from decoder_sc import (
    upper_llr,
    lower_llr,
    _active_llr_level,
    _active_bit_level,
    _frozen_indices,
    sc_decode,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _pm_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if int(bit) == hard else abs(llr)


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, L, B, pm=0.0):
        self.L = L
        self.B = B
        self.pm = pm

    def copy(self):
        return _Path(self.L.copy(), self.B.copy(), self.pm)


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 LLR/比特平面）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = _frozen_indices(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(
            sorted(set(range(N)) - self.frozen_set), dtype=int
        )

    def _update_llrs(self, path, l):
        L, B = path.L, path.B
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        B = path.B
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            from decoder_sc import sc_decode

            frozen_bits = np.zeros(self.N, dtype=bool)
            for idx in self.frozen_set:
                frozen_bits[idx] = True
            u_hat = sc_decode(llr_ch, frozen_bits)
            return u_hat, 0.0

        L0 = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B0 = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        L0[:, 0] = llr_ch
        paths = [_Path(L0, B0, 0.0)]

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm += _pm_penalty(llr, 0)
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm += _pm_penalty(llr, bit)
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        best = paths[0]

        if self.crc_length > 0 and len(self.info_indices) >= self.crc_length:
            valid = []
            for p in paths:
                info = p.B[:, self.n].astype(int)[self.info_indices]
                if crc_check(info, self.crc_length):
                    valid.append(p)
            if valid:
                best = valid[0]

        u_hat = best.B[:, self.n].astype(int)
        return u_hat, best.pm


def scl_equals_sc(N, frozen_bits, trials=20):
    """单路径 SCL 应等价于 SC"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    rng = np.random.default_rng(0)
    for _ in range(trials):
        llr = rng.normal(0, 2, N)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            return False
    return True
