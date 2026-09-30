"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_nat):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_nat
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        from encoder import bit_reversal_permutation

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_nat = llr_ch[br]

        paths = [_Path(self.N, self.n, llr_nat)]

        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr0 = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr0, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    _update_bits(path.B, l, self.n, self.N)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        p2 = _Path.__new__(_Path)
                        p2.pm = path.pm + self._pm_penalty(llr0, u)
                        p2.L = path.L.copy()
                        p2.B = path.B.copy()
                        p2.u_hat = path.u_hat.copy()
                        p2.u_hat[l] = u
                        p2.B[l, self.n] = u
                        _update_bits(p2.B, l, self.n, self.N)
                        new_paths.append(p2)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_bits = np.where(~self.frozen_bits)[0]
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[info_bits], self.crc_length)
            ]
            if valid:
                paths = valid
        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
