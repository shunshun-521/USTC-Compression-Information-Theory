"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    payload = bits[:-crc_length]
    return np.array_equal(crc_encode(payload, crc_length)[-crc_length:], bits[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（路径复制实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs_path(self, path, l):
        L, B = path.L, path.B
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits_path(self, path, l):
        B = path.B
        n = self.n
        if l < self.N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, u):
        v = 0 if llr >= 0 else 1
        return 0.0 if u == v else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch.copy())]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            new_paths = []
            for path in paths:
                self._update_llrs_path(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    pen = self._pm_penalty(llr, 0)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits_path(path, l)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        p = _Path(self.N, self.n, path.L[:, 0].copy())
                        p.L[:, :] = path.L
                        p.B[:, :] = path.B
                        p.pm = path.pm + self._pm_penalty(llr, u)
                        p.B[l, self.n] = u
                        p.u_hat = path.u_hat.copy()
                        p.u_hat[l] = u
                        self._update_bits_path(p, l)
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            if valid:
                paths = valid
        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
