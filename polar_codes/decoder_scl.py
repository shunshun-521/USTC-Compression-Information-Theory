"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from channel import prepare_channel_llr
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)


POLY8 = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
POLY16 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=int)


def _crc_divide(info_bits, poly):
    bits = np.concatenate([info_bits.astype(int), np.zeros(len(poly) - 1, dtype=int)])
    for i in range(len(info_bits)):
        if bits[i]:
            bits[i:i + len(poly)] ^= poly
    return bits[-(len(poly) - 1):]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = POLY8 if crc_length == 8 else POLY16
    remainder = _crc_divide(info_bits, poly)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = POLY8 if crc_length == 8 else POLY16
    return np.all(_crc_divide(bits, poly) == 0)


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr

    def copy(self):
        new = _Path(len(self.L), self.L.shape[1] - 1, self.L[:, 0])
        new.pm = self.pm
        new.L = self.L.copy()
        new.B = self.B.copy()
        return new


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _pm_penalty(self, llr, u_bit):
        preferred = 0 if llr >= 0 else 1
        return 0.0 if u_bit == preferred else abs(llr)

    def decode(self, llr_ch):
        llr = prepare_channel_llr(llr_ch, self.N)
        paths = [_Path(self.N, self.n, llr)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr_phi = path.L[l, self.n]

                if self.frozen_bits[l]:
                    new_p = path.copy()
                    new_p.pm += self._pm_penalty(llr_phi, 0)
                    new_p.B[l, self.n] = 0
                    _update_bits(new_p.B, l, self.n, self.N)
                    new_paths.append(new_p)
                else:
                    for u_bit in (0, 1):
                        new_p = path.copy()
                        new_p.pm += self._pm_penalty(llr_phi, u_bit)
                        new_p.B[l, self.n] = u_bit
                        _update_bits(new_p.B, l, self.n, self.N)
                        new_paths.append(new_p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        best = None
        if self.crc_length > 0:
            crc_pass = []
            for p in paths:
                u = p.B[:, self.n].astype(int)
                info_bits = u[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            if crc_pass:
                best = min(crc_pass, key=lambda p: p.pm)

        if best is None:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, self.n].astype(int), best.pm
