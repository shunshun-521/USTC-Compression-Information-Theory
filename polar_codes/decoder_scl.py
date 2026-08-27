"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_penalty(self, llr_val, u):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u == hard else abs(llr_val)

    def _update_llrs(self, paths, l):
        for path in paths:
            for s in range(self.n - _active_llr_level(l, self.n), self.n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                    else:
                        path.L[j, s + 1] = g_operation(
                            path.L[j - branch_size, s],
                            path.L[j, s],
                            path.B[j - branch_size, s + 1],
                        )

    def _update_bits(self, paths, l):
        if l < self.N // 2:
            return
        for path in paths:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                        path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        class PathState:
            __slots__ = ("L", "B", "pm", "u_hat", "copied")

            def __init__(self, parent=None):
                if parent is None:
                    self.L = np.zeros((self_N, self_n + 1), dtype=np.float64)
                    self.B = np.zeros((self_N, self_n + 1), dtype=np.int8)
                    self.L[:, 0] = llr_ch
                    self.pm = 0.0
                    self.u_hat = np.zeros(self_N, dtype=np.int8)
                    self.copied = True
                else:
                    self.L = parent.L
                    self.B = parent.B
                    self.pm = parent.pm
                    self.u_hat = parent.u_hat.copy()
                    self.copied = False

            def ensure_copy(self):
                if not self.copied:
                    self.L = self.L.copy()
                    self.B = self.B.copy()
                    self.copied = True

        self_N = self.N
        self_n = self.n
        paths = [PathState()]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            self._update_llrs(paths, l)

            candidates = []
            for path in paths:
                llr_val = path.L[l, self.n]
                if self.frozen_bits[l]:
                    new_path = PathState(path)
                    new_path.ensure_copy()
                    new_path.pm += self._path_metric_penalty(llr_val, 0)
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = PathState(path)
                        new_path.ensure_copy()
                        new_path.pm += self._path_metric_penalty(llr_val, u)
                        new_path.u_hat[l] = u
                        new_path.B[l, self.n] = u
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]
            self._update_bits(paths, l)

        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[info_positions], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
