"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    sc_decode,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = _Path(self.L.shape[0], self.L.shape[1] - 1, self.L[:, 0])
        p.pm = self.pm
        p.L[:] = self.L
        p.B[:] = self.B
        p.u_hat[:] = self.u_hat
        return p


def _llr_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if hard == bit else abs(llr)


def _update_llrs(path, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
            else:
                top_bit = path.B[j - branch_size, s + 1]
                path.L[j, s + 1] = g_operation(
                    path.L[j - branch_size, s], path.L[j, s], top_bit
                )
    return path.L[l, n]


def _update_bits(path, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                    path.B[j - branch_size, s]
                )
                path.B[j, s - 1] = path.B[j, s]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """返回最优路径的 u_hat 及路径度量 pm。"""
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr = np.asarray(llr_ch, dtype=np.float64)
        llr = llr[bit_reversal_permutation(self.N)]
        paths = [_Path(self.N, self.n, llr)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                llr_val = _update_llrs(path, l, self.n, self.N)
                if l in self.frozen_set:
                    p = path.copy()
                    p.pm += _llr_penalty(llr_val, 0)
                    p.B[l, self.n] = 0
                    p.u_hat[l] = 0
                    _update_bits(p, l, self.n, self.N)
                    candidates.append(p)
                else:
                    for bit in (0, 1):
                        p = path.copy()
                        p.pm += _llr_penalty(llr_val, bit)
                        p.B[l, self.n] = bit
                        p.u_hat[l] = bit
                        _update_bits(p, l, self.n, self.N)
                        candidates.append(p)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
