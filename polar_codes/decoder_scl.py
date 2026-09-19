"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _reorder_llrs,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（MSB-first）。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_polynomial(crc_length)
    reg = 0
    for bit in info_bits:
        fb = ((reg >> (crc_length - 1)) & 1) ^ bit
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
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_polynomial(crc_length)
    reg = 0
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) & 1) ^ bit
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if fb:
            reg ^= poly
    return reg == 0


def _hard_bit_from_llr(llr):
    return 0 if llr >= 0 else 1


def _path_metric_update(pm, llr, u):
    hard = _hard_bit_from_llr(llr)
    if u != hard:
        return pm + abs(llr)
    return pm


class _Path:
    __slots__ = ("pm", "u_hat", "l_mat", "c_mat")

    def __init__(self, n_len, n):
        self.pm = 0.0
        self.u_hat = np.zeros(n_len, dtype=np.int8)
        self.l_mat = np.full((n_len, n + 1), np.nan, dtype=np.float64)
        self.c_mat = np.full((n_len, n + 1), np.nan, dtype=np.float64)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = (
            np.asarray(info_indices, dtype=int)
            if info_indices is not None
            else np.where(~self.frozen_bits)[0]
        )

    def _compute_llr(self, path, phi):
        l_idx = _bit_reversed(phi, self.n)
        for s in range(self.n - _active_llr_level(l_idx, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l_idx, self.N, block_size):
                if j % block_size < branch_size:
                    path.l_mat[j, s + 1] = f_operation(
                        path.l_mat[j, s], path.l_mat[j + branch_size, s]
                    )
                else:
                    path.l_mat[j, s + 1] = g_operation(
                        path.l_mat[j - branch_size, s],
                        path.l_mat[j, s],
                        path.c_mat[j - branch_size, s + 1],
                    )
        return path.l_mat[l_idx, self.n]

    def _propagate_bit(self, path, phi, u_bit):
        l_idx = _bit_reversed(phi, self.n)
        path.c_mat[l_idx, self.n] = u_bit
        if l_idx < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l_idx, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    path.c_mat[j - branch_size, s - 1] = int(path.c_mat[j, s]) ^ int(
                        path.c_mat[j - branch_size, s]
                    )
                    path.c_mat[j, s - 1] = path.c_mat[j, s]

    def decode(self, llr_ch):
        llr = _reorder_llrs(llr_ch)
        n_len = self.N
        n = self.n

        paths = [_Path(n_len, n)]
        paths[0].l_mat[:, 0] = llr

        for phi in range(n_len):
            l_idx = _bit_reversed(phi, n)
            candidates = []

            for path in paths:
                llr_bit = self._compute_llr(path, phi)
                branches = [0] if l_idx in self.frozen_set else [0, 1]
                for u_bit in branches:
                    new_path = _Path(n_len, n)
                    new_path.l_mat[:] = path.l_mat
                    new_path.c_mat[:] = path.c_mat
                    new_path.u_hat[:] = path.u_hat
                    new_path.pm = _path_metric_update(path.pm, llr_bit, u_bit)
                    new_path.u_hat[l_idx] = u_bit
                    self._propagate_bit(new_path, phi, u_bit)
                    candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = valid[0] if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy().astype(int), best.pm
