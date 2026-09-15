"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _lower_llr_minsum,
    f_operation,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_divide(data_bits, poly, crc_length):
    reg = [0] * crc_length
    for bit in np.asarray(data_bits, dtype=int):
        feedback = bit ^ reg[0]
        reg = reg[1:] + [0]
        if feedback:
            for i in range(crc_length):
                if (poly >> (crc_length - 1 - i)) & 1:
                    reg[i] ^= feedback
    return np.array(reg, dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return np.concatenate([info_bits, _crc_divide(info_bits, poly, crc_length)])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class _SCLPath:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)

    def clone(self):
        p = _SCLPath.__new__(_SCLPath)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_set = sorted(set(range(N)) - self.frozen_set)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = int(path.B[j - branch_size, s + 1]) if not np.isnan(path.B[j - branch_size, s + 1]) else 0
                    path.L[j, s + 1] = _lower_llr_minsum(path.L[j, s], path.L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _pm_add(llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_SCLPath(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    child = path.clone()
                    child.pm += self._pm_add(llr_val, 0)
                    child.B[l, self.n] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = path.clone()
                        child.pm += self._pm_add(llr_val, bit)
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            crc_ok = [p for p in paths if crc_check(
                np.nan_to_num(p.B[:, self.n], nan=0).astype(int)[self.info_set],
                self.crc_length,
            )]
            best = min(crc_ok if crc_ok else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        u_hat = np.nan_to_num(best.B[:, self.n], nan=0).astype(int)
        return u_hat, best.pm
