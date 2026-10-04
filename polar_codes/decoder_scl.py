"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Permuted SCD 结构
"""
import math
import numpy as np
from decoder_sc import (
    bit_reversed,
    _active_bit_level,
    _active_llr_level,
    _lower_llr,
    _map_channel_llr,
    _upper_llr,
)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    mask = (1 << crc_length) - 1
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _pm_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return abs(llr) if u != u_hard else 0.0


class _Path:
    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch

    def copy(self):
        p = _Path(len(self.u_hat), int(math.log2(len(self.u_hat))), self.L[:, 0])
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        p.L = self.L.copy()
        p.B = self.B.copy()
        return p

    def update_llrs(self, l, n):
        N = len(self.u_hat)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j, s], self.L[j - branch_size, s], int(top_bit)
                    )

    def update_bits(self, l, n):
        N = len(self.u_hat)
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=np.int64) if info_indices is not None else None
        )
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def _crc_ok(self, u_hat):
        if self.crc_length <= 0:
            return True
        if self.info_indices is not None:
            payload = u_hat[self.info_indices]
        else:
            payload = u_hat[~self.frozen_bits]
        return crc_check(payload, self.crc_length)

    def decode(self, llr_ch):
        llr_ch = _map_channel_llr(np.asarray(llr_ch, dtype=np.float64))
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                path.update_llrs(l, self.n)
                llr_bit = path.L[l, self.n]
                if l in self.frozen_set:
                    child = path.copy()
                    child.pm += _pm_penalty(llr_bit, 0)
                    child.u_hat[l] = 0
                    child.B[l, self.n] = 0
                    child.update_bits(l, self.n)
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm += _pm_penalty(llr_bit, u)
                        child.u_hat[l] = u
                        child.B[l, self.n] = u
                        child.update_bits(l, self.n)
                        candidates.append(child)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_ok(p.u_hat)]
            best = valid[0] if valid else paths[0]
        else:
            best = paths[0]
        return best.u_hat.astype(int), best.pm
