"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_step(reg, bit, poly, crc_length):
    fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
    reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ (fb * poly)
    return reg


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in info_bits:
        reg = _crc_step(reg, b, poly, crc_length)
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in bits:
        reg = _crc_step(reg, b, poly, crc_length)
    return reg == 0


class _PathState:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（路径复制实现，列表大小 L）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.Lsize = list_size
        self.crc_length = crc_length
        self.decode_order = [
            int(bit_reversal_permutation(N)[i]) for i in range(N)
        ]

    def _update_llrs(self, path, l):
        L, B = path.L, path.B
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 1 << (s + 1)
            half = block >> 1
            for j in range(l, self.N, block):
                if j % block < half:
                    L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - half, s], L[j, s], int(B[j - half, s + 1])
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        B = path.B
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 1 << s
            half = block >> 1
            for j in range(l, -1, -block):
                if j % block >= half:
                    B[j - half, s - 1] = (B[j, s] + B[j - half, s]) % 2
                    B[j, s - 1] = B[j, s]

    def _branch_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            for path in paths:
                self._update_llrs(path, l)

            llr_l = paths[0].L[l, self.n]
            new_paths = []

            if self.frozen_bits[l]:
                for path in paths:
                    path.pm += self._branch_penalty(path.L[l, self.n], 0)
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
            else:
                for path in paths:
                    llr = path.L[l, self.n]
                    for bit in (0, 1):
                        child = _PathState(self.N, self.n, llr_ch)
                        child.L[:] = path.L
                        child.B[:] = path.B
                        child.u_hat[:] = path.u_hat
                        child.pm = path.pm + self._branch_penalty(llr, bit)
                        child.B[l, self.n] = bit
                        child.u_hat[l] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.Lsize]

        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            info_bits = best.u_hat[~self.frozen_bits]
            if crc_check(info_bits, self.crc_length):
                return best.u_hat.astype(int), best.pm
            for p in sorted(paths, key=lambda x: x.pm):
                info_bits = p.u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    return p.u_hat.astype(int), p.pm

        return best.u_hat.astype(int), best.pm
