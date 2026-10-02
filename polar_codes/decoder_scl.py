"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np

from decoder_sc import (
    _SCDState,
    _bit_reversed,
    sc_decode,
)


def _crc8_bits(info_bits):
    poly = 0x07
    reg = 0
    for b in np.asarray(info_bits, dtype=int):
        reg ^= b << 7
        for _ in range(8):
            if reg & 0x80:
                reg = ((reg << 1) ^ poly) & 0xFF
            else:
                reg = (reg << 1) & 0xFF
    return [(reg >> i) & 1 for i in range(7, -1, -1)]


def _crc16_bits(info_bits):
    poly = 0x8005
    reg = 0
    for b in np.asarray(info_bits, dtype=int):
        reg ^= b << 15
        for _ in range(8):
            if reg & 0x8000:
                reg = ((reg << 1) ^ poly) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return [(reg >> i) & 1 for i in range(15, -1, -1)]


def crc_encode(info_bits, crc_length=8):
    bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        crc = _crc8_bits(bits)
    elif crc_length == 16:
        crc = _crc16_bits(bits)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([bits, np.asarray(crc, dtype=int)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length=crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _SCLPath:
    def __init__(self, state, pm=0.0):
        self.state = state
        self.pm = pm

    def clone(self):
        return _SCLPath(copy.deepcopy(self.state), self.pm)

    @staticmethod
    def pm_penalty(llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [_SCLPath(_SCDState(llr_ch, self.frozen_bits), 0.0)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                path.state._update_llrs(l)
                llr_bit = path.state.L[l, self.n]
                if l in self.frozen_set:
                    child = path.clone()
                    child.pm += _SCLPath.pm_penalty(llr_bit, 0)
                    child.state.B[l, self.n] = 0
                    child.state._update_bits(l)
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        child = path.clone()
                        child.pm += _SCLPath.pm_penalty(llr_bit, u)
                        child.state.B[l, self.n] = u
                        child.state._update_bits(l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = None
        if self.crc_length > 0:
            passed = []
            for p in paths:
                u_hat = p.state.B[:, self.n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    passed.append(p)
            if passed:
                best = min(passed, key=lambda p: p.pm)
        if best is None:
            best = min(paths, key=lambda p: p.pm)

        u_hat = best.state.B[:, self.n].astype(int)
        return u_hat, best.pm
