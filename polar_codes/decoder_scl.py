"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    bit_llr_at_phi,
    _frozen_to_br_domain,
    _EPS,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_update(reg, bit, width, poly):
    mask = (1 << width) - 1
    fb = ((reg >> (width - 1)) ^ int(bit)) & 1
    return ((reg << 1) & mask) ^ (poly * fb)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly, width = CRC8_POLY, 8
    elif crc_length == 16:
        poly, width = CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for bit in info_bits:
        reg = _crc_update(reg, bit, width, poly)
    crc_bits = np.array(
        [(reg >> (width - 1 - i)) & 1 for i in range(width)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly, width = CRC8_POLY, 8
    elif crc_length == 16:
        poly, width = CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for bit in bits:
        reg = _crc_update(reg, bit, width, poly)
    return reg == 0


class Path:
    __slots__ = ("llr", "u_hat", "pm")

    def __init__(self, N):
        self.llr = None
        self.u_hat = np.zeros(N, dtype=int)
        self.pm = 0.0


class SCLDecoder:
    """SCL 译码器（递归路径扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N

        if self.list_size == 1:
            from decoder_sc import sc_decode_nonrecursive

            return sc_decode_nonrecursive(llr_ch, self.frozen_bits), 0.0

        rev = bit_reversal_permutation(N)
        frozen_br = _frozen_to_br_domain(self.frozen_bits, N)

        paths = [Path(N)]
        paths[0].llr = llr_ch.copy()

        for phi in range(N):
            new_paths = []
            for path in paths:
                llr0 = bit_llr_at_phi(path.llr, path.u_hat, phi, self.N)
                if abs(llr0) < _EPS:
                    llr0 = float(llr_ch[phi])

                if frozen_br[phi]:
                    p = Path(N)
                    p.llr = path.llr
                    p.u_hat = path.u_hat.copy()
                    p.pm = path.pm + self._pm_penalty(llr0, 0)
                    p.u_hat[phi] = 0
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = Path(N)
                        p.llr = path.llr
                        p.u_hat = path.u_hat.copy()
                        p.pm = path.pm + self._pm_penalty(llr0, bit)
                        p.u_hat[phi] = bit
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            ok_paths = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            paths = ok_paths if ok_paths else paths

        best = min(paths, key=lambda p: p.pm)
        u_br = best.u_hat.copy()
        u_hat = np.zeros(N, dtype=int)
        u_hat[rev] = u_br
        return u_hat, best.pm

