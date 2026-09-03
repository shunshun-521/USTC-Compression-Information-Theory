"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Permuted SCD
"""
import math
import numpy as np
from encoder import bit_reversal_index
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


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
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _pm_penalty(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch=None):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        if llr_ch is not None:
            self.L[:, 0] = llr_ch

    def clone(self):
        p = _Path.__new__(_Path)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Permuted SCD + 路径分裂）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.decode_order = [bit_reversal_index(i, self.n) for i in range(N)]
        self.info_mask = ~self.frozen_bits

    def _update_llrs(self, path, l):
        N, n = self.N, self.n
        L, B = path.L, path.B
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

    def _update_bits(self, path, l):
        N, n = self.N, self.n
        B = path.B
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L_size = self.N, self.n, self.list_size

        paths = [_Path(N, n, llr_ch)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, n]
                if l in self.frozen_set:
                    path.pm += _pm_penalty(llr, 0)
                    path.B[l, n] = 0
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for u_bit in (0, 1):
                        branch = path.clone()
                        branch.pm += _pm_penalty(llr, u_bit)
                        branch.B[l, n] = u_bit
                        self._update_bits(branch, l)
                        candidates.append(branch)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:L_size]

        if self.crc_length > 0:
            passed = [
                p for p in paths
                if crc_check(p.B[:, n][self.info_mask], self.crc_length)
            ]
            best = min(passed, key=lambda p: p.pm) if passed else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, n].astype(int).copy(), best.pm
