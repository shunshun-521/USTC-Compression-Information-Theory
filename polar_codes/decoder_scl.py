"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from scd_core import (
    scd,
    bit_reversed,
    active_llr_level,
    active_bit_level,
    upper_ref,
    lower_llr,
    _b,
)
from encoder import bit_reversal_permutation


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=np.uint8)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.uint8)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _align_llr(self, llr_ch):
        br = bit_reversal_permutation(self.N)
        return np.asarray(llr_ch, dtype=np.float64)[br]

    def _advance_path(self, path, l):
        N, n = self.N, self.n
        for s in range(n - active_llr_level(l, n), n):
            bs = 2 ** (s + 1)
            brs = bs // 2
            for j in range(l, N, bs):
                if j % bs < brs:
                    path.L[j, s + 1] = upper_ref(path.L[j, s], path.L[j + brs, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s], path.L[j - brs, s], _b(path.B[j - brs, s + 1])
                    )
        return path.L[l, n]

    def _propagate_bits(self, path, l):
        N, n = self.N, self.n
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            bs = 2 ** s
            brs = bs // 2
            for j in range(l, -1, -bs):
                if j % bs >= brs:
                    path.B[j - brs, s - 1] = _b(path.B[j, s]) ^ _b(path.B[j - brs, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr = self._align_llr(llr_ch)
        N, n = self.N, self.n
        paths = [_Path(N, n, llr.copy())]

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []
            for path in paths:
                llr_bit = self._advance_path(path, l)
                if l in self.frozen_set:
                    penalty = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    path.pm += penalty
                    path.u_hat[l] = 0
                    path.B[l, n] = 0
                    self._propagate_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p2 = _Path(N, n, path.L[:, 0].copy())
                        p2.L = path.L.copy()
                        p2.B = path.B.copy()
                        p2.u_hat = path.u_hat.copy()
                        p2.pm = path.pm
                        p2.u_hat[l] = bit
                        p2.B[l, n] = bit
                        if bit == (0 if llr_bit >= 0 else 1):
                            pass
                        else:
                            p2.pm += abs(llr_bit)
                        self._propagate_bits(p2, l)
                        new_paths.append(p2)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            chosen = min(valid or paths, key=lambda p: p.pm)
        else:
            chosen = paths[0]
        return chosen.u_hat.copy(), chosen.pm
