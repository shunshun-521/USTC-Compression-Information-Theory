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
    sc_decode,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _PathState:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


def _update_llrs(path, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
            else:
                path.L[j, s + 1] = _lower_llr(
                    path.L[j, s],
                    path.L[j - branch_size, s],
                    int(path.B[j - branch_size, s + 1]),
                )


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


def _pm_add(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    return pm + (0.0 if u == u_hard else abs(llr))


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [_PathState(self.N, self.n, llr_ch)]
        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                _update_llrs(path, l, self.n, self.N)
                llr = path.L[l, self.n]
                if self.frozen_bits[l]:
                    path.pm = _pm_add(path.pm, llr, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    _update_bits(path, l, self.n, self.N)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        cp = _PathState(self.N, self.n, llr_ch)
                        cp.L[:] = path.L
                        cp.B[:] = path.B
                        cp.u_hat[:] = path.u_hat
                        cp.pm = _pm_add(path.pm, llr, u)
                        cp.u_hat[l] = u
                        cp.B[l, self.n] = u
                        _update_bits(cp, l, self.n, self.N)
                        new_paths.append(cp)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
