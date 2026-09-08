"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation, bit_reversed_index
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    sc_decode,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(info_bits, poly, crc_length):
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(bits[:-crc_length], poly, crc_length)
    expected = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.array_equal(bits[-crc_length:], expected)


def _path_metric_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_perm):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.L[:, 0] = llr_perm
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def fork(self):
        new_path = _Path.__new__(_Path)
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


def _update_llrs(path, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
            else:
                top_bit = int(path.B[j - branch_size, s + 1])
                path.L[j, s + 1] = g_operation(path.L[j - branch_size, s], path.L[j, s], top_bit)


def _update_bits(path, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                path.B[j, s - 1] = path.B[j, s]


class SCLDecoder:
    """SCL 译码器（路径分裂时复制数组，Lazy Copy 语义）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_positions = np.where(~self.frozen_bits)[0]
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size <= 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_perm = llr_ch[self.br]
        paths = [_Path(self.N, self.n, llr_perm)]

        for phi in range(self.N):
            l = bit_reversed_index(phi, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path, l, self.n, self.N)
                llr = path.L[l, self.n]

                if phi in self.frozen_set:
                    new_path = path.fork()
                    new_path.pm += _path_metric_penalty(llr, 0)
                    new_path.u_hat[phi] = 0
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path, l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.fork()
                        new_path.pm += _path_metric_penalty(llr, bit)
                        new_path.u_hat[phi] = bit
                        new_path.B[l, self.n] = bit
                        _update_bits(new_path, l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_positions], self.crc_length)]
            chosen = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            chosen = min(paths, key=lambda p: p.pm)

        return chosen.u_hat.copy(), chosen.pm
