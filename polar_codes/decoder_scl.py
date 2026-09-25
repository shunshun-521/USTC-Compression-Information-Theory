"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
)


_CRC8_GEN = [1, 0, 0, 0, 0, 0, 1, 1, 1]          # x^8 + x^2 + x + 1
_CRC16_GEN = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # CRC-16-IBM


def _crc_generator(crc_length):
    if crc_length == 8:
        return _CRC8_GEN
    if crc_length == 16:
        return _CRC16_GEN
    raise ValueError("crc_length must be 8 or 16")


def _crc_divide(msg_bits, crc_length):
    """多项式长除法计算 CRC 余数。"""
    gen = _crc_generator(crc_length)
    msg = list(map(int, msg_bits)) + [0] * crc_length
    for i in range(len(msg_bits)):
        if msg[i] == 1:
            for j in range(len(gen)):
                if i + j < len(msg):
                    msg[i + j] ^= gen[j]
    return np.array(msg[len(msg_bits):], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    crc_bits = _crc_divide(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    gen = _crc_generator(crc_length)
    msg = list(map(int, bits)) + [0] * crc_length
    for i in range(len(bits)):
        if msg[i] == 1:
            for j in range(len(gen)):
                if i + j < len(msg):
                    msg[i + j] ^= gen[j]
    return sum(msg[len(bits):]) == 0


class _Path:
    __slots__ = ("pm", "B", "parent", "branch_bit")

    def __init__(self, N, n):
        self.pm = 0.0
        self.B = np.full((N, n + 1), np.nan)
        self.parent = None
        self.branch_bit = None


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = frozen_bits.astype(bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _get_bit(self, path, i, s):
        val = path.B[i, s]
        if np.isnan(val):
            if path.parent is not None:
                val = self._get_bit(path.parent, i, s)
            else:
                val = 0
        return int(val)

    def _set_bit(self, path, i, s, val):
        if np.isnan(path.B[i, s]):
            path.B[i, s] = val
        else:
            child = _Path(self.N, self.n)
            child.B = path.B.copy()
            child.pm = path.pm
            child.parent = path.parent
            child.B[i, s] = val
            return child
        return path

    def _update_llrs(self, paths, L, l):
        for path in paths:
            for s in range(self.n - _active_llr_level(l, self.n), self.n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                    else:
                        top_bit = self._get_bit(path, j - branch_size, s + 1)
                        L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    b_j = self._get_bit(path, j, s)
                    b_top = self._get_bit(path, j - branch_size, s)
                    path.B[j - branch_size, s - 1] = b_j ^ b_top
                    path.B[j, s - 1] = b_j

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        L[:, 0] = llr_ch

        paths = [_Path(self.N, self.n)]

        for phi, l in enumerate(self.decode_order):
            self._update_llrs(paths, L, l)
            llr = L[l, self.n]
            new_paths = []

            if l in self.frozen_set:
                for path in paths:
                    path.B[l, self.n] = 0
                    path.pm += self._path_metric_penalty(llr, 0)
                    self._update_bits(path, l)
                    new_paths.append(path)
            else:
                for path in paths:
                    for bit in (0, 1):
                        child = _Path(self.N, self.n)
                        child.B = path.B.copy()
                        child.pm = path.pm + self._path_metric_penalty(llr, bit)
                        child.B[l, self.n] = bit
                        child.parent = path
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best_crc = None
        best_any = paths[0]
        for path in paths:
            if path.pm < best_any.pm:
                best_any = path
            if self.crc_length > 0:
                u = path.B[:, self.n].astype(int)
                info_mask = ~self.frozen_bits
                info_bits = u[info_mask]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path.pm < best_crc.pm:
                        best_crc = path

        chosen = best_crc if best_crc is not None else best_any
        u_hat = chosen.B[:, self.n].astype(int)
        for i in range(self.N):
            if np.isnan(chosen.B[i, self.n]):
                u_hat[i] = self._get_bit(chosen, i, self.n)
        return u_hat, chosen.pm
