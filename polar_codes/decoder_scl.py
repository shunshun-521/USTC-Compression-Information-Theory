"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed,
    upper_llr,
    lower_llr,
    active_llr_level,
    active_bit_level,
    sc_decode,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _PathState:
    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = _PathState.__new__(_PathState)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


def _update_llrs_path(path, l, n, N):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
            else:
                top_bit = path.B[j - branch_size, s + 1]
                path.L[j, s + 1] = lower_llr(
                    path.L[j, s], path.L[j - branch_size, s], top_bit
                )


def _update_bits_path(path, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                    path.B[j - branch_size, s]
                )
                path.B[j, s - 1] = path.B[j, s]


def _pm_add(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                _update_llrs_path(path, l, self.n, self.N)
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    pm = path.pm + _pm_add(llr, 0)
                    candidates.append((pm, path, 0))
                else:
                    for u in (0, 1):
                        pm = path.pm + _pm_add(llr, u)
                        candidates.append((pm, path, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.L]

            new_paths = []
            for pm, parent, u in candidates:
                child = parent.copy()
                child.pm = pm
                child.u_hat[l] = u
                child.B[l, self.n] = u
                _update_bits_path(child, l, self.n, self.N)
                new_paths.append(child)
            paths = new_paths

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            info_pos = np.where(self.frozen_bits == 0)[0]
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[info_pos], self.crc_length)
            ]
            if valid:
                paths = valid
        best = paths[0]
        return best.u_hat.copy(), best.pm
