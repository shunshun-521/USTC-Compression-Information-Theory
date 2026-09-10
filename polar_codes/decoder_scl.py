"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _lower_llr,
    _upper_llr,
    bit_reversed_index,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


CRC_GENERATORS = {
    8: [1, 0, 0, 0, 0, 0, 1, 1, 1],
    16: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
}


def _crc_remainder_bits(bits, generator):
    r = len(generator) - 1
    msg = [int(b) for b in bits] + [0] * r
    for i in range(len(bits)):
        if msg[i]:
            for j in range(len(generator)):
                msg[i + j] ^= generator[j]
    return np.array(msg[-r:], dtype=np.int8)


def _crc_verify_bits(bits, generator):
    r = len(generator) - 1
    msg = [int(b) for b in bits]
    for i in range(len(bits) - r):
        if msg[i]:
            for j in range(len(generator)):
                msg[i + j] ^= generator[j]
    return all(x == 0 for x in msg[-r:])


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    generator = CRC_GENERATORS[crc_length]
    crc_bits = _crc_remainder_bits(info_bits, generator)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    generator = CRC_GENERATORS[crc_length]
    return _crc_verify_bits(bits, generator)


class _SCLPath:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Permuted SCL，Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)

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
                    path.L[j, s + 1] = _upper_llr(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = llr_ch[self.rev]

        paths = [_SCLPath(self.N, self.n) for _ in range(self.list_size)]
        active = [paths[0]]
        active[0].L[:, 0] = llr_perm
        active[0].pm = 0.0

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            is_frozen = l in self.frozen_set
            candidates = []

            for p_idx, path in enumerate(active):
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if is_frozen:
                    candidates.append(
                        (path.pm + self._pm_penalty(llr, 0), p_idx, 0)
                    )
                else:
                    for bit in (0, 1):
                        candidates.append(
                            (path.pm + self._pm_penalty(llr, bit), p_idx, bit)
                        )

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_active = []
            used = set()
            for pm_new, src_idx, bit in candidates:
                if len(new_active) >= self.list_size:
                    break
                if src_idx in used:
                    dst = _SCLPath(self.N, self.n)
                    self._copy_path(active[src_idx], dst)
                else:
                    dst = active[src_idx]
                    used.add(src_idx)

                dst.pm = pm_new
                dst.B[l, self.n] = bit
                dst.u_hat[l] = bit
                self._update_bits(dst, l)
                new_active.append(dst)

            active = new_active

        best_crc_path = None
        best_crc_pm = float("inf")
        best_path = None
        best_pm = float("inf")

        for path in active:
            if path.pm < best_pm:
                best_pm = path.pm
                best_path = path
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if path.pm < best_crc_pm:
                        best_crc_pm = path.pm
                        best_crc_path = path

        chosen = best_crc_path if best_crc_path is not None else best_path
        return chosen.u_hat.astype(int), chosen.pm
