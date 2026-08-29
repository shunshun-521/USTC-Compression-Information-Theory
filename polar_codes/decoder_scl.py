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
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x107
    if crc_length == 16:
        return 0x11021
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class _Path:
    __slots__ = ("pm", "u_hat", "L", "B", "active")

    def __init__(self, n, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=int)
            if info_indices is not None
            else np.where(frozen_bits == 0)[0]
        )
        self.br_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    def _update_llrs(self, path, l):
        n, N = self.n, self.N
        L, B = path.L, path.B
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )
        return L[l, n]

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        n, N = self.n, self.N
        B = path.B
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [_Path(n, N)]
        paths[0].L[:, 0] = llr_ch.copy()

        for l in self.br_order:
            is_frozen = self.frozen_bits[l] == 1
            new_paths = []

            for path in paths:
                if not path.active:
                    continue
                llr = self._update_llrs(path, l)
                if is_frozen:
                    p = _Path(n, N)
                    p.pm = path.pm + self._pm_penalty(llr, 0)
                    p.u_hat = path.u_hat.copy()
                    p.L = path.L.copy()
                    p.B = path.B.copy()
                    p.u_hat[l] = 0
                    p.B[l, n] = 0
                    self._update_bits(p, l)
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = _Path(n, N)
                        p.pm = path.pm + self._pm_penalty(llr, bit)
                        p.u_hat = path.u_hat.copy()
                        p.L = path.L.copy()
                        p.B = path.B.copy()
                        p.u_hat[l] = bit
                        p.B[l, n] = bit
                        self._update_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat, best.pm
