"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _prepare_llr,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = 0x07 if crc_length == 8 else 0x8005

    reg = 0
    for bit in info_bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if fb:
            reg ^= poly

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 crc_length 位是否为正确 CRC。"""
    if crc_length <= 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    poly = 0x07 if crc_length == 8 else 0x8005

    reg = 0
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if fb:
            reg ^= poly
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n, llr_natural):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_natural
        self.pm = 0.0

    def copy(self):
        new = _Path.__new__(_Path)
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.pm = self.pm
        return new


class SCLDecoder:
    """SCL 译码器（Permuted SC + 路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [int(bit_reversal_permutation(N)[i]) for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l, bit):
        path.B[l, self.n] = bit
        if l >= self.N // 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size >> 1
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                            path.B[j - branch_size, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_natural = _prepare_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr_natural)]

        for l in self.decode_order:
            for path in paths:
                self._update_llrs(path, l)

            new_paths = []
            for path in paths:
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr, 0)
                    self._update_bits(path, l, 0)
                    new_paths.append(path)
                else:
                    p0 = path.copy()
                    p1 = path.copy()
                    p0.pm += self._pm_penalty(llr, 0)
                    p1.pm += self._pm_penalty(llr, 1)
                    self._update_bits(p0, l, 0)
                    self._update_bits(p1, l, 1)
                    new_paths.extend([p0, p1])

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        candidates = paths
        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            valid = []
            for p in candidates:
                u = p.B[:, self.n].astype(int)
                info_bits = u[info_mask]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                candidates = valid

        best = min(candidates, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm
