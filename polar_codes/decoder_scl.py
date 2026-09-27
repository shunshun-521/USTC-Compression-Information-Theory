"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    f_operation,
    _map_channel_llrs,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_mod_bits(bits, crc_length):
    """对 bits 做模 2 长除法，poly 含隐式 x^crc_length 项"""
    poly = _crc_poly(crc_length)
    msg = np.asarray(bits, dtype=np.int8).copy().ravel()
    n = len(msg)
    for i in range(n - crc_length + 1):
        if msg[i]:
            for j in range(crc_length + 1):
                if j == 0:
                    continue
                tap = (poly >> (crc_length - j)) & 1 if j <= crc_length else 0
                if j == crc_length:
                    tap = 1
                if tap:
                    msg[i + j] ^= 1
    return msg[-crc_length:]


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    rem = _crc_mod_bits(padded, crc_length)
    return np.concatenate([info_bits, rem])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8).ravel()
    rem = _crc_mod_bits(bits, crc_length)
    return np.all(rem == 0)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, n, N, llr_mapped):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_mapped
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        p = _Path.__new__(_Path)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    b = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = (
                        path.L[j, s] + path.L[j - branch_size, s]
                        if b == 0
                        else path.L[j, s] - path.L[j - branch_size, s]
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_mapped = _map_channel_llrs(llr_ch)
        paths = [_Path(self.n, self.N, llr_mapped)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            expanded = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    u = 0
                    pen = abs(llr) if llr < 0 else 0.0
                    child = path.copy()
                    child.pm += pen
                    child.u_hat[l] = u
                    child.B[l, self.n] = u
                    self._update_bits(child, l)
                    expanded.append(child)
                else:
                    for u in (0, 1):
                        expected = 0 if llr >= 0 else 1
                        pen = 0.0 if u == expected else abs(llr)
                        child = path.copy()
                        child.pm += pen
                        child.u_hat[l] = u
                        child.B[l, self.n] = u
                        self._update_bits(child, l)
                        expanded.append(child)

            expanded.sort(key=lambda p: p.pm)
            paths = expanded[: self.L]

        if self.crc_length > 0:
            valid = []
            for i, p in enumerate(paths):
                info_bits = p.u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    valid.append(i)
            idx = valid[np.argmin([paths[i].pm for i in valid])] if valid else int(
                np.argmin([p.pm for p in paths])
            )
        else:
            idx = int(np.argmin([p.pm for p in paths]))

        return paths[idx].u_hat, paths[idx].pm
