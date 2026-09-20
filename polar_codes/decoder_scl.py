"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _as_frozen_mask,
    f_operation,
    g_operation,
    precompute_sc_indices,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_register(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    for b in bits:
        reg = ((reg << 1) | int(b)) & mask
        if reg & msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_register(msg, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_register(bits, poly, crc_length) == 0


class _PathState:
    __slots__ = ("L", "B", "u_hat", "pm")

    def __init__(self, N, n, channel):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = channel
        self.u_hat = np.zeros(N, dtype=int)
        self.pm = 0.0

    def copy(self):
        other = _PathState(self.L.shape[0], self.L.shape[1] - 1, self.L[:, 0])
        other.L[:] = self.L
        other.B[:] = self.B
        other.u_hat[:] = self.u_hat
        other.pm = self.pm
        return other


def _update_llrs_path(L, B, l, n):
    N = L.shape[0]
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)


def _update_bits_path(B, l, n):
    N = B.shape[0]
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = _as_frozen_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)
        precompute_sc_indices(N)

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        channel = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.N, self.n, channel)]

        for i in range(self.N):
            l = self.br[i]
            new_paths = []
            for path in paths:
                _update_llrs_path(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    child = path.copy()
                    child.pm += self._pm_penalty(llr, 0)
                    child.B[l, self.n] = 0
                    child.u_hat[l] = 0
                    _update_bits_path(child.B, l, self.n)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm += self._pm_penalty(llr, u)
                        child.B[l, self.n] = u
                        child.u_hat[l] = u
                        _update_bits_path(child.B, l, self.n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best_crc = None
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path.pm < best_crc.pm:
                        best_crc = path

        best = best_crc if best_crc is not None else paths[0]
        return best.u_hat.copy(), best.pm
