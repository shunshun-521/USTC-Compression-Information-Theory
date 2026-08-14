"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation, g_operation, _bit_reversed, _active_llr_level, _active_bit_level,
)


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    remainder = _crc_remainder(bits[:-crc_length], crc_length)
    received = 0
    for bit in bits[-crc_length:]:
        received = (received << 1) | int(bit)
    return remainder == received


class PathState:
    __slots__ = ('pm', 'L', 'B', 'L_src', 'B_src')

    def __init__(self, n, N, llr):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.L[:, 0] = llr
        self.L_src = self.L
        self.B_src = self.B

    def fork(self):
        child = PathState.__new__(PathState)
        child.pm = self.pm
        child.L = None
        child.B = None
        child.L_src = self.L if self.L is not None else self.L_src
        child.B_src = self.B if self.B is not None else self.B_src
        return child

    def materialize(self, n, N):
        if self.L is None:
            self.L = self.L_src.copy()
            self.B = self.B_src.copy()
            self.L_src = self.L
            self.B_src = self.B


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            bs = 1 << (s + 1)
            br = bs // 2
            for j in range(l, self.N, bs):
                if j % bs < br:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + br, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j, s], path.L[j - br, s], path.B[j - br, s + 1]
                    )

    def _propagate_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            bs = 1 << s
            br = bs // 2
            for j in range(l, -1, -bs):
                if j % bs >= br:
                    path.B[j - br, s - 1] = path.B[j, s] ^ path.B[j - br, s]
                    path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _pm_add(pm, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return pm + (0.0 if u_bit == hard else abs(llr_val))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        paths = [PathState(n, N, llr_ch)]

        for phi in range(N):
            l = _bit_reversed(phi, n)
            new_paths = []
            for path in paths:
                path.materialize(n, N)
                self._update_llrs(path, l)
                llr_val = path.L[l, n]
                if self.frozen_bits[l]:
                    child = path.fork()
                    child.materialize(n, N)
                    child.pm = self._pm_add(path.pm, llr_val, 0)
                    child.B[l, n] = 0
                    self._propagate_bits(child, l)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        child = path.fork()
                        child.materialize(n, N)
                        child.pm = self._pm_add(path.pm, llr_val, u_bit)
                        child.B[l, n] = u_bit
                        self._propagate_bits(child, l)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = min(paths, key=lambda p: p.pm)
        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            u_hat[i] = best.B[i, n]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                path.materialize(n, N)
                info_bits = path.B[self.info_indices, n]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p.pm)
                for i in range(N):
                    u_hat[i] = best.B[i, n]

        return u_hat.astype(int), best.pm
