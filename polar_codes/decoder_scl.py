"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    lower_llr,
    upper_llr,
)


def _crc_remainder(bits, poly, crc_len):
    reg = np.zeros(crc_len, dtype=np.int8)
    for b in bits:
        feedback = b ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            for i in range(crc_len):
                if (poly >> (crc_len - 1 - i)) & 1:
                    reg[i] ^= feedback
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    return np.concatenate([info_bits, rem])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits, poly, crc_length)
    return np.all(rem == 0)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Permuted SCD 扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
        self.L = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(np.asarray(frozen_bits, dtype=int) == 0)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []
            for pidx, path in enumerate(paths):
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen:
                    candidates.append((path.pm + self._penalty(llr, 0), pidx, 0))
                else:
                    candidates.append((path.pm + self._penalty(llr, 0), pidx, 0))
                    candidates.append((path.pm + self._penalty(llr, 1), pidx, 1))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.L]

            new_paths = []
            for pm, pidx, u in candidates:
                parent = paths[pidx]
                child = _Path(self.N, self.n)
                child.pm = pm
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()
                child.B[l, self.n] = 0 if l in self.frozen else u
                child.u_hat[l] = child.B[l, self.n]
                self._update_bits(child, l)
                new_paths.append(child)
            paths = new_paths

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    return path.u_hat.astype(int), path.pm
        best = paths[0]
        return best.u_hat.astype(int), best.pm
