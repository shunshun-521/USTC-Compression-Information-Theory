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
    _channel_llr_to_decoder,
    _sc_decode_core,
    f_operation,
    g_operation,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _PathState:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            L, B = path.L, path.B
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        B = path.B
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr = _channel_llr_to_decoder(llr_ch)
        paths = [_PathState(self.N, self.n, llr)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path.L[l, self.n]
                if l in self.frozen_set:
                    child = _PathState(self.N, self.n, llr)
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.pm = path.pm + self._pm_penalty(llr_bit, 0)
                    child.B[l, self.n] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = _PathState(self.N, self.n, llr)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = path.pm + self._pm_penalty(llr_bit, u)
                        child.B[l, self.n] = u
                        self._update_bits(child, l)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        u_hat_all = np.array([p.B[:, self.n].astype(int) for p in paths])
        pms = np.array([p.pm for p in paths])

        if self.crc_length > 0:
            valid = []
            for idx, u_hat in enumerate(u_hat_all):
                payload = u_hat[self.info_idx]
                if crc_check(payload, self.crc_length):
                    valid.append((pms[idx], u_hat))
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        best = int(np.argmin(pms))
        return u_hat_all[best], float(pms[best])
