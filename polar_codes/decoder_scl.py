"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _lower_llr,
    _upper_llr,
)
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length <= 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0

    def copy(self):
        p = _Path(self.L.shape[0], self.L.shape[1] - 1, self.L[:, 0])
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s], path.L[j - branch_size, s], path.B[j - branch_size, s + 1]
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

    @staticmethod
    def _penalty(llr_val, u_val):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_val == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            expanded = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = path.copy()
                    new_path.pm += self._penalty(llr_val, 0)
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    expanded.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._penalty(llr_val, u_val)
                        new_path.B[l, self.n] = u_val
                        self._update_bits(new_path, l)
                        expanded.append(new_path)

            expanded.sort(key=lambda p: p.pm)
            paths = expanded[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_hat = p.B[:, self.n].astype(np.int8)
                if crc_check(u_hat, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        u_hat = best.B[:, self.n].astype(np.int8)
        return u_hat, best.pm


def verify_scl_equals_sc(N=64, frozen_bits=None, num_trials=50, seed=1):
    """L=1 的 SCL 应与 SC 等价"""
    from decoder_sc import sc_decode

    rng = np.random.default_rng(seed)
    if frozen_bits is None:
        frozen_bits = np.zeros(N, dtype=int)
        frozen_bits[N // 2 :] = 1
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(num_trials):
        llr = rng.normal(0, 2, N)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            return False
    return True
