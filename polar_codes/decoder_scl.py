"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _SCDCore,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _permute_channel_llr,
    _upper_llr,
    sc_decode,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class _SCLPath:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制，SCD 调度）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _copy_path(self, src, dst):
        dst.L[:] = src.L
        dst.B[:] = src.B
        dst.pm = src.pm
        dst.u_hat[:] = src.u_hat

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = _permute_channel_llr(llr_ch, self.N)
        paths = [_SCLPath(self.N, self.n) for _ in range(self.list_size)]
        paths[0].L[:, 0] = llr_perm
        active = 1

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for pidx in range(active):
                path = paths[pidx]
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(llr, 0)
                    path.pm += pen
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    candidates.append((path.pm, pidx, None))
                else:
                    for bit in (0, 1):
                        candidates.append((path.pm + self._pm_penalty(llr, bit), pidx, bit))

            candidates.sort(key=lambda x: x[0])
            new_paths = [_SCLPath(self.N, self.n) for _ in range(self.list_size)]
            new_active = 0

            for pm, src_idx, bit in candidates:
                if new_active >= self.list_size:
                    break
                self._copy_path(paths[src_idx], new_paths[new_active])
                new_paths[new_active].pm = pm
                if bit is not None:
                    new_paths[new_active].u_hat[l] = bit
                    new_paths[new_active].B[l, self.n] = bit
                    self._update_bits(new_paths[new_active], l)
                new_active += 1

            paths = new_paths
            active = new_active

        best_pm = float("inf")
        best_u = paths[0].u_hat.copy()
        crc_pass = []

        for pidx in range(active):
            u = paths[pidx].u_hat
            pm = paths[pidx].pm
            if self.crc_length > 0:
                info_bits = u[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append((pm, u))
            if pm < best_pm:
                best_pm = pm
                best_u = u.copy()

        if crc_pass:
            crc_pass.sort(key=lambda x: x[0])
            return crc_pass[0][1], crc_pass[0][0]
        return best_u, best_pm


def scl_decode_equivalent_sc(llr_ch, frozen_bits):
    u_scl, _ = SCLDecoder(len(llr_ch), frozen_bits, list_size=1).decode(llr_ch)
    u_sc = sc_decode(llr_ch, frozen_bits)
    return u_scl, u_sc
