"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import sc_decode
from scd_vendor.decoder_utils import (
    active_bit_level,
    active_llr_level,
    hard_decision,
    lower_llr,
    upper_llr,
)
from scd_vendor.utils import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, degree):
    reg = 0
    for b in bits:
        reg ^= int(b) << (degree - 1)
        if reg & (1 << (degree - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << degree) - 1)
        else:
            reg = (reg << 1) & ((1 << degree) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly, deg = CRC8_POLY, 8
    elif crc_length == 16:
        poly, deg = CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, deg)
    crc_bits = np.array([(rem >> (deg - 1 - i)) & 1 for i in range(deg)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly, deg = CRC8_POLY, 8
    elif crc_length == 16:
        poly, deg = CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, deg) == 0


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch

    def copy(self):
        cp = _Path.__new__(_Path)
        cp.pm = self.pm
        cp.L = self.L.copy()
        cp.B = self.B.copy()
        return cp

    def update_llrs(self, l, n):
        for s in range(n - active_llr_level(l, n), n):
            block_size = int(2 ** (s + 1))
            branch_size = block_size // 2
            for j in range(l, self.L.shape[0], block_size):
                if j % block_size < branch_size:
                    top_llr = self.L[j, s]
                    btm_llr = self.L[j + branch_size, s]
                    self.L[j, s + 1] = upper_llr(top_llr, btm_llr)
                else:
                    btm_llr = self.L[j, s]
                    top_llr = self.L[j - branch_size, s]
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = lower_llr(btm_llr, top_llr, top_bit)

    def update_bits(self, l, n, N):
        if l < N / 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = int(2 ** s)
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def u_hat(self, n):
        return self.B[:, n].astype(np.int8)


class SCLDecoder:
    """SCL 译码器（基于 vendor SC 因子图更新）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.frozen = set(int(i) for i in np.where(self.frozen_bits.astype(bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=np.int64) if info_indices is not None else None
        )

    def _pm_update(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [_Path(N, n, llr_ch)]
        decode_order = [bit_reversed(i, n) for i in range(N)]

        for l in decode_order:
            for path in paths:
                path.update_llrs(l, n)

            llr_l = paths[0].L[l, n]
            new_paths = []

            if l in self.frozen:
                for path in paths:
                    path.pm = self._pm_update(path.pm, llr_l, 0)
                    path.B[l, n] = 0
                    path.update_bits(l, n, N)
                    new_paths.append(path)
            else:
                for path in paths:
                    for u in (0, 1):
                        cp = path.copy()
                        cp.pm = self._pm_update(path.pm, llr_l, u)
                        cp.B[l, n] = u
                        cp.update_bits(l, n, N)
                        new_paths.append(cp)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            idx = self.info_indices
            if idx is None:
                idx = np.where(~self.frozen_bits.astype(bool))[0]
            crc_ok = [p for p in paths if crc_check(p.u_hat(n)[idx], self.crc_length)]
            best = crc_ok[0] if crc_ok else paths[0]
        else:
            best = paths[0]

        return best.u_hat(n), best.pm
