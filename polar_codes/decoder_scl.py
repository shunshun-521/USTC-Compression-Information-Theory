"""
极化码 SCL（串行抵消列表）译码器，含 CRC 辅助 CA-SCL
"""
import math
import numpy as np

from decoder_sc import (
    _bit_reversed,
    _update_bits,
    _update_llr,
    f_operation,
    g_operation,
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

    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits.astype(int), crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _llr_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]

        paths = [_Path(self.N, self.n)]
        paths[0].L[:, self.n] = llr

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                _update_llr(path.L, path.B, l, self.n)
                llr_bit = path.L[l, 0]

                if self.frozen_bits[l]:
                    pen = self._llr_penalty(llr_bit, 0)
                    path.pm += pen
                    path.u_hat[l] = 0
                    path.B[l, 0] = 0
                    _update_bits(path.B, l, self.n)
                    new_paths.append(path)
                else:
                    for u_cand in (0, 1):
                        p2 = _Path(self.N, self.n)
                        p2.L = path.L.copy()
                        p2.B = path.B.copy()
                        p2.u_hat = path.u_hat.copy()
                        p2.pm = path.pm + self._llr_penalty(llr_bit, u_cand)
                        p2.u_hat[l] = u_cand
                        p2.B[l, 0] = u_cand
                        _update_bits(p2.B, l, self.n)
                        new_paths.append(p2)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_idx = np.where(~self.frozen_bits)[0]
                payload = p.u_hat[info_idx]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
