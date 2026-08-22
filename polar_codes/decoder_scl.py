"""
极化码 SCL（串行抵消列表）译码器
"""
import math
import os
import sys

import numpy as np

_REF = os.path.join(os.path.dirname(__file__), "polar_ref")
if _REF not in sys.path:
    sys.path.insert(0, _REF)

from SCD import SCD  # noqa: E402


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = 0x07 if crc_length == 8 else (0x8005 if crc_length == 16 else None)
    if poly is None:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    reg = 0
    for bit in info_bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _bit_reversed(x, n):
    r = 0
    for i in range(n):
        if x & (1 << i):
            r |= 1 << (n - 1 - i)
    return r


class _PathSCD(SCD):
    def __init__(self, myPC, llr_ch):
        myPC.likelihoods = llr_ch
        super().__init__(myPC)
        self.pm = 0.0
        self.u = np.zeros(self.myPC.N, dtype=int)

    def clone(self):
        p = _PathSCD.__new__(_PathSCD)
        p.myPC = self.myPC
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u = self.u.copy()
        return p


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _make_pc(self):
        class PC:
            pass
        pc = PC()
        pc.N = self.N
        pc.n = self.n
        pc.frozen = list(self.frozen_set)
        return pc

    def decode(self, llr_ch):
        if self.list_size == 1:
            from decoder_sc import sc_decode
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        pc = self._make_pc()
        paths = [_PathSCD(pc, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            new_paths = []
            for path in paths:
                path.update_llrs(l)
                llr_bit = path.L[l, self.n]
                if l in self.frozen_set:
                    np_ = path.clone()
                    if llr_bit < 0:
                        np_.pm += abs(llr_bit)
                    np_.u[l] = 0
                    np_.B[l, self.n] = 0
                    np_.update_bits(l)
                    new_paths.append(np_)
                else:
                    for bit in (0, 1):
                        np_ = path.clone()
                        exp = 0 if llr_bit >= 0 else 1
                        if bit != exp:
                            np_.pm += abs(llr_bit)
                        np_.u[l] = bit
                        np_.B[l, self.n] = bit
                        np_.update_bits(l)
                        new_paths.append(np_)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        valid = [p for p in paths if self._crc_ok(p.u)]
        best = min(valid or paths, key=lambda p: p.pm)
        return best.u.copy(), best.pm

    def _crc_ok(self, u):
        if self.crc_length == 0:
            return True
        return crc_check(u[self.info_positions], self.crc_length)
