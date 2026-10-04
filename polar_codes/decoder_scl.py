"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversed
from decoder_sc import (
    _SCDCore,
    _frozen_indices,
    active_bit_level,
    active_llr_level,
    lower_llr,
    upper_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（基于 Permuted SCD，路径分裂时复制 L/B）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = _frozen_indices(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~np.asarray(frozen_bits).astype(bool))[0]

    @staticmethod
    def _pm_penalty(llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _clone_core(self, core):
        new = _SCDCore(self.N, self.frozen_set)
        new.L = np.array(core.L, copy=True)
        new.B = np.array(core.B, copy=True)
        return new

    def _update_llrs(self, core, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    core.L[j, s + 1] = upper_llr(core.L[j, s], core.L[j + branch_size, s])
                else:
                    core.L[j, s + 1] = lower_llr(
                        core.L[j, s],
                        core.L[j - branch_size, s],
                        int(core.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, core, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    core.B[j - branch_size, s - 1] = int(core.B[j, s]) ^ int(
                        core.B[j - branch_size, s]
                    )
                    core.B[j, s - 1] = core.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        core0 = _SCDCore(self.N, self.frozen_set)
        core0.set_channel(llr_ch)
        paths = [(0.0, core0)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []
            for pm, core in paths:
                self._update_llrs(core, l)
                llr = core.L[l, self.n]
                if l in self.frozen_set:
                    child = self._clone_core(core)
                    child.B[l, self.n] = 0
                    self._update_bits(child, l)
                    candidates.append((pm + self._pm_penalty(llr, 0), child))
                else:
                    for u in (0, 1):
                        child = self._clone_core(core)
                        child.B[l, self.n] = u
                        self._update_bits(child, l)
                        candidates.append((pm + self._pm_penalty(llr, u), child))

            candidates.sort(key=lambda x: x[0])
            paths = candidates[: self.list_size]

        best_pm, best_core = min(paths, key=lambda x: x[0])
        u_hat = best_core.B[:, self.n].astype(np.int8)

        if self.crc_length > 0:
            valid = []
            for pm, core in paths:
                u = core.B[:, self.n].astype(np.int8)
                if crc_check(u[self.info_indices], self.crc_length):
                    valid.append((pm, u))
            if valid:
                best_pm, u_hat = min(valid, key=lambda x: x[0])

        return u_hat, best_pm
