"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    bit_reversal_permutation,
    bit_reversed,
    active_llr_level,
    active_bit_level,
    precompute_sc_indices,
)


CRC_POLYNOMIALS = {
    8: np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int),
    16: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int),
}


def _crc_remainder(bits, poly):
    reg = np.zeros(len(poly) - 1, dtype=int)
    for bit in bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg ^= poly[1:]
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(info_bits, poly)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(bits, poly)
    return np.all(remainder == 0)


class PathState:
    __slots__ = ("L", "B", "pm", "u_hat", "active")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True

    def copy(self):
        new_path = PathState(self.L.shape[0], self.L.shape[1] - 1)
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


def _update_llrs_path(path, l, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        N = path.L.shape[0]
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
            else:
                top_bit = path.B[j - branch_size, s + 1]
                path.L[j, s + 1] = g_operation(path.L[j - branch_size, s], path.L[j, s], top_bit)


def _update_bits_path(path, l, n):
    N = path.L.shape[0]
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                path.B[j, s - 1] = path.B[j, s]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = precompute_sc_indices(N)
        self.rev = bit_reversal_permutation(N)

    def _pm_update(self, pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit == hard:
            return pm
        return pm + abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [PathState(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch[self.rev]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                _update_llrs_path(path, l, self.n)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    child = path.copy()
                    child.pm = self._pm_update(path.pm, llr, 0)
                    child.B[l, self.n] = 0
                    child.u_hat[l] = 0
                    _update_bits_path(child, l, self.n)
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm = self._pm_update(path.pm, llr, bit)
                        child.B[l, self.n] = bit
                        child.u_hat[l] = bit
                        _update_bits_path(child, l, self.n)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
