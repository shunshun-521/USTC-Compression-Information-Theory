"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation, g_operation, precompute_sc_indices,
    _bit_reversed, _active_llr_level, _active_bit_level,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_register(bits, crc_length):
    """串行 LFSR CRC，每次输入 1 比特"""
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = _crc_register(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    return _crc_register(bits, crc_length) == 0


class _SCLPath:
    __slots__ = ('pm', 'u_hat', 'L', 'B')

    def __init__(self, N, n, llr=None):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        if llr is not None:
            self.L[:, 0] = llr


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)

    def _compute_llr(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], path.B[j - branch_size, s + 1]
                    )
        return path.L[l, self.n]

    def _update_bits(self, path, l, u_val):
        path.u_hat[l] = u_val
        path.B[l, self.n] = u_val
        if l >= self.N // 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = (
                            path.B[j, s] + path.B[j - branch_size, s]
                        ) % 2
                        path.B[j, s - 1] = path.B[j, s]

    def _clone(self, path):
        child = _SCLPath(self.N, self.n)
        child.pm = path.pm
        child.u_hat = path.u_hat.copy()
        child.L = path.L.copy()
        child.B = path.B.copy()
        return child

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.rev].copy()
        paths = [_SCLPath(self.N, self.n, llr)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                llr_root = self._compute_llr(path, l)
                if self.frozen_bits[l]:
                    child = self._clone(path)
                    if llr_root < 0:
                        child.pm += abs(llr_root)
                    self._update_bits(child, l, 0)
                    candidates.append(child)
                else:
                    for u_cand in (0, 1):
                        child = self._clone(path)
                        hard = 0 if llr_root >= 0 else 1
                        if u_cand != hard:
                            child.pm += abs(llr_root)
                        self._update_bits(child, l, u_cand)
                        candidates.append(child)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
