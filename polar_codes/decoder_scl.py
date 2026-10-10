"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed_index,
    f_operation,
    g_operation,
    precompute_sc_indices,
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
    mask = 1 << (crc_length - 1)
    for b in info_bits:
        msb = (reg & mask) != 0
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if int(b):
            reg ^= mask
        if msb:
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = 1 << (crc_length - 1)
    for b in bits:
        msb = (reg & mask) != 0
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if int(b):
            reg ^= mask
        if msb:
            reg ^= poly
    return reg == 0


def _update_llrs(L, B, l, n, N):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = _Path(self.L.shape[0], self.L.shape[1] - 1, self.L[:, 0])
        p.pm = self.pm
        p.L[:] = self.L
        p.B[:] = self.B
        p.u_hat[:] = self.u_hat
        return p


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        precompute_sc_indices(N)

    @staticmethod
    def _pm_update(pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]
        decode_order = [bit_reversed_index(i, self.n) for i in range(self.N)]

        for l in decode_order:
            candidates = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr_bit = path.L[l, self.n]
                if l in self.frozen_set:
                    p = path.copy()
                    p.pm = self._pm_update(p.pm, llr_bit, 0)
                    p.B[l, self.n] = 0
                    p.u_hat[l] = 0
                    _update_bits(p.B, l, self.n, self.N)
                    candidates.append(p)
                else:
                    for u in (0, 1):
                        p = path.copy()
                        p.pm = self._pm_update(p.pm, llr_bit, u)
                        p.B[l, self.n] = u
                        p.u_hat[l] = u
                        _update_bits(p.B, l, self.n, self.N)
                        candidates.append(p)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.L]

        if self.crc_length > 0:
            passed = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(passed if passed else paths, key=lambda p: p.pm)
        else:
            best = paths[0]
        return best.u_hat, best.pm
