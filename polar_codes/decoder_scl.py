"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    f_operation,
    g_operation,
)
from encoder import bit_reversed_index


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = _crc_poly(crc_length)
    reg = 0
    bits = np.asarray(info_bits, dtype=int)
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) & ((1 << crc_length) - 1)) | bit
        if msb:
            reg ^= poly
    for _ in range(crc_length):
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if msb:
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


def _path_update_llrs(path, l, n):
    start = n - _active_llr_level(l, n)
    for s in range(start, n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, path.L.shape[0], block_size):
            if j % block_size < branch_size:
                top = path.L[j, s]
                bottom = path.L[j + branch_size, s]
                path.L[j, s + 1] = f_operation(top, bottom)
            else:
                bottom = path.L[j, s]
                top = path.L[j - branch_size, s]
                top_bit = path.B[j - branch_size, s + 1]
                path.L[j, s + 1] = g_operation(top, bottom, top_bit)


def _path_update_bits(path, l, n, N):
    if l < N // 2:
        return
    end = n - _active_bit_level(l, n)
    for s in range(n, end, -1):
        block_size = 1 << s
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
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = [Path(N, n)]
        paths[0].L[:, 0] = llr_ch

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                _path_update_llrs(path, l, n)
                llr_l = path.L[l, n]

                if self.frozen_bits[l]:
                    child = Path(N, n)
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.pm = path.pm + self._pm_penalty(llr_l, 0)
                    child.u_hat = path.u_hat.copy()
                    child.B[l, n] = 0
                    child.u_hat[l] = 0
                    _path_update_bits(child, l, n, N)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = Path(N, n)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = path.pm + self._pm_penalty(llr_l, bit)
                        child.u_hat = path.u_hat.copy()
                        child.B[l, n] = bit
                        child.u_hat[l] = bit
                        _path_update_bits(child, l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid or paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
