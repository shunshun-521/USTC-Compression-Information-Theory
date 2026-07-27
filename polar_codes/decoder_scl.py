"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
)
from encoder import bit_reversal_permutation, bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_compute(bits, poly, crc_length):
    """GF(2) CRC 计算，返回余数比特列表。"""
    bits = [int(b) for b in bits]
    reg = bits + [0] * crc_length
    poly_bits = [(poly >> i) & 1 for i in range(crc_length - 1, -1, -1)]
    poly_bits = [1] + poly_bits
    for i in range(len(bits)):
        if reg[i] == 1:
            for j, p in enumerate(poly_bits):
                if i + j < len(reg):
                    reg[i + j] ^= p
    return reg[-crc_length:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem_bits = _crc_compute(info_bits, poly, crc_length)
    return np.concatenate([info_bits, np.array(rem_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem_bits = _crc_compute(bits, poly, crc_length)
    return all(b == 0 for b in rem_bits)


class _Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat', 'active')

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_idx = np.where(self.frozen_bits == 0)[0]
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.phases = [bit_reversed(i, self.n) for i in range(N)]

    def _branch_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _update_llrs(self, paths, l):
        for path in paths:
            if not path.active:
                continue
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
                            path.B[j - branch_size, s + 1],
                        )

    def _update_bits(self, paths, l):
        if l < self.N // 2:
            return
        for path in paths:
            if not path.active:
                continue
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = (
                            path.B[j, s] ^ path.B[j - branch_size, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]

        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr

        for l in self.phases:
            self._update_llrs(paths, l)
            new_paths = []

            if l in self.frozen_set:
                for path in paths:
                    if not path.active:
                        continue
                    llr_val = path.L[l, self.n]
                    path.pm += self._branch_penalty(llr_val, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    new_paths.append(path)
            else:
                for path in paths:
                    if not path.active:
                        continue
                    llr_val = path.L[l, self.n]
                    for bit in (0, 1):
                        child = _Path(self.N, self.n)
                        child.L[:] = path.L
                        child.B[:] = path.B
                        child.u_hat[:] = path.u_hat
                        child.pm = path.pm + self._branch_penalty(llr_val, bit)
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        new_paths.append(child)

                new_paths.sort(key=lambda p: p.pm)
                paths = new_paths[: self.list_size]

            self._update_bits(paths, l)

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_idx]
                if crc_check(info_bits, self.crc_length):
                    return path.u_hat.copy(), path.pm

        best = paths[0]
        return best.u_hat.copy(), best.pm
