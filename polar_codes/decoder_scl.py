"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np

from _scd_core import PermutedSCD, bit_reversed
from decoder_sc import sc_decode


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _pm_penalty(llr, u):
    u_hat = 0 if llr >= 0 else 1
    return 0.0 if u == u_hat else abs(llr)


class _Path:
    def __init__(self, N, llr):
        self.scd = PermutedSCD(N, llr)
        self.pm = 0.0


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen = set(np.where(self.frozen_bits)[0].tolist())
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [_Path(self.N, llr_ch.copy())]
        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                path.scd.update_llrs(l)
                llr = path.scd.L[l, self.n]
                if l in self.frozen:
                    cp = copy.deepcopy(path)
                    cp.pm += _pm_penalty(llr, 0)
                    cp.scd.B[l, cp.scd.n] = 0
                    cp.scd.update_bits(l)
                    new_paths.append(cp)
                else:
                    for u in (0, 1):
                        cp = copy.deepcopy(path)
                        cp.pm += _pm_penalty(llr, u)
                        cp.scd.B[l, cp.scd.n] = u
                        cp.scd.update_bits(l)
                        new_paths.append(cp)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            for p in paths:
                u_hat = p.scd.B[:, p.scd.n].astype(int)
                if crc_check(u_hat, self.crc_length):
                    return u_hat, p.pm
        best = paths[0]
        return best.scd.B[:, best.scd.n].astype(int), best.pm
