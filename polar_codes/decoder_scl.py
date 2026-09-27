"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_scalar,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_init):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_init


class SCLDecoder:
    """SCL 译码器（路径级复制 L/B 阵列）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    @staticmethod
    def _penalty(llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_init = llr_ch[self.br]
        paths = [_Path(self.N, self.n, llr_init.copy())]

        for i in range(self.N):
            l = _bit_reversed_scalar(i, self.n)
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr0 = path.L[l, self.n]
                if self.frozen_bits[l]:
                    path.pm += self._penalty(llr0, 0)
                    path.B[l, self.n] = 0
                    _update_bits(path.B, l, self.n)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        cp = _Path(self.N, self.n, path.L[:, 0])
                        cp.L[:] = path.L
                        cp.B[:] = path.B
                        cp.pm = path.pm + self._penalty(llr0, u)
                        cp.B[l, self.n] = u
                        _update_bits(cp.B, l, self.n)
                        new_paths.append(cp)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best_crc = None
        best_pm = None
        for path in paths:
            u_hat = path.B[:, self.n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path.pm < best_crc.pm:
                        best_crc = path
            if best_pm is None or path.pm < best_pm:
                best_pm = path

        chosen = best_crc if best_crc is not None else best_pm
        return chosen.B[:, self.n].astype(int), chosen.pm
