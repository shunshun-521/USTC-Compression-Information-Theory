"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _SCDCore
from encoder import bit_reversal_index, bit_reversal_permutation


def _crc8_bits(info_bits):
    poly = 0x07
    reg = 0
    for b in np.asarray(info_bits, dtype=int):
        reg ^= int(b) << 7
        for _ in range(8):
            reg = ((reg << 1) ^ poly) & 0xFF if (reg & 0x80) else (reg << 1) & 0xFF
    return np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)


def _crc16_bits(info_bits):
    poly = 0x8005
    reg = 0
    for b in np.asarray(info_bits, dtype=int):
        reg ^= int(b) << 15
        for _ in range(16):
            reg = ((reg << 1) ^ poly) & 0xFFFF if (reg & 0x8000) else (reg << 1) & 0xFFFF
    return np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        return np.concatenate([info_bits, _crc8_bits(info_bits)])
    if crc_length == 16:
        return np.concatenate([info_bits, _crc16_bits(info_bits)])
    raise ValueError("crc_length must be 8 or 16")


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        return np.array_equal(_crc8_bits(bits[:-8]), bits[-8:])
    if crc_length == 16:
        return np.array_equal(_crc16_bits(bits[:-16]), bits[-16:])
    return False


class _Path:
    __slots__ = ("core", "pm", "u_hat")

    def __init__(self, N, frozen_set, llr_perm):
        self.core = _SCDCore(N, np.zeros(N, dtype=bool))
        self.core.frozen = frozen_set
        self.core.set_channel(llr_perm)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_set = set(np.where(~self.frozen_bits)[0])
        self.L = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    @staticmethod
    def _pm_update(pm, llr, u_bit):
        u_hard = 0 if llr >= 0 else 1
        if u_bit != u_hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_perm = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = [_Path(self.N, self.frozen_set, llr_perm)]

        for i in range(self.N):
            l = bit_reversal_index(i, self.n)
            candidates = []
            for path in paths:
                path.core.update_llrs(l)
                llr = path.core.L[l, self.n]
                if l in self.frozen_set:
                    path.pm = self._pm_update(path.pm, llr, 0)
                    path.core.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    path.core.update_bits(l)
                    candidates.append(path)
                else:
                    for u_bit in (0, 1):
                        child = _Path(self.N, self.frozen_set, llr_perm)
                        child.core.L = path.core.L.copy()
                        child.core.B = path.core.B.copy()
                        child.u_hat = path.u_hat.copy()
                        child.pm = self._pm_update(path.pm, llr, u_bit)
                        child.core.B[l, self.n] = u_bit
                        child.u_hat[l] = u_bit
                        child.core.update_bits(l)
                        candidates.append(child)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.L]

        best = paths[0]
        if self.crc_length > 0:
            info_idx = sorted(self.info_set)
            passed = [p for p in paths if crc_check(p.u_hat[info_idx], self.crc_length)]
            if passed:
                best = min(passed, key=lambda p: p.pm)
        return best.u_hat.astype(int), float(best.pm)
