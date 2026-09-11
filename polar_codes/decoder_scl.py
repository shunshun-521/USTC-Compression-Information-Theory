"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
    _prepare_frozen,
)
from encoder import bit_reversed


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_polynomial(crc_length)
    msg = 0
    for bit in bits:
        msg = (msg << 1) | int(bit)
    msg <<= crc_length
    divisor = poly | (1 << crc_length)
    for i in range(len(bits) + crc_length - 1, crc_length - 1, -1):
        if (msg >> i) & 1:
            msg ^= divisor << (i - crc_length)
    return msg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    return _crc_remainder(bits, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _prepare_frozen(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    @staticmethod
    def _llr_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    top = j - branch_size
                    path.L[j, s + 1] = g_operation(
                        path.L[top, s], path.L[j, s], path.B[top, s + 1]
                    )

    def _update_bits(self, path, l, u_val):
        path.B[l, self.n] = u_val
        path.u_hat[l] = u_val
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _clone_path(self, src):
        dst = _Path(self.N, self.n, src.L[:, 0])
        dst.pm = src.pm
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.u_hat = src.u_hat.copy()
        return dst

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in [bit_reversed(i, self.n) for i in range(self.N)]:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen:
                    new_path = self._clone_path(path)
                    new_path.pm += self._llr_penalty(llr, 0)
                    self._update_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = self._clone_path(path)
                        new_path.pm += self._llr_penalty(llr, u)
                        self._update_bits(new_path, l, u)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            chosen = (
                min(valid, key=lambda p: p.pm)
                if valid
                else min(paths, key=lambda p: p.pm)
            )
        else:
            chosen = min(paths, key=lambda p: p.pm)

        return chosen.u_hat.astype(int), chosen.pm
