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
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_len):
    """计算 CRC 余数。"""
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_len - 1))
        for _ in range(crc_len):
            if reg & (1 << (crc_len - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_len) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_len) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    data = bits[:-crc_length]
    expected = crc_encode(data, crc_length)
    return np.array_equal(bits, expected)


class Path:
    """单条 SCL 译码路径。"""

    __slots__ = ('pm', 'L', 'C', 'u_hat')

    def __init__(self, n, N):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.C = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _update_llr(self, path, l):
        """更新路径 LLR。"""
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.C[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        """比特回传。"""
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.C[j - branch_size, s - 1] = (
                        path.C[j, s] + path.C[j - branch_size, s]
                    ) % 2
                    path.C[j, s - 1] = path.C[j, s]

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.n, self.N)]
        rev = bit_reversal_permutation(self.N)
        paths[0].L[:, 0] = llr_ch[rev]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llr(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr_val, 0)
                    path.u_hat[l] = 0
                    path.C[l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = Path(self.n, self.N)
                        child.pm = path.pm + self._pm_penalty(llr_val, u_bit)
                        child.L = path.L.copy()
                        child.C = path.C.copy()
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[l] = u_bit
                        child.C[l, self.n] = u_bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.u_hat, best.pm
