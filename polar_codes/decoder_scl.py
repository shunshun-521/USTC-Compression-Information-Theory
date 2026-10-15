"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import (
    _SCDEngine,
    bit_reversed,
    hard_decision,
)
from encoder import bit_reversal_permutation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=int).ravel()
    pad = (8 - len(bits) % 8) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=int)])
    out = bytearray()
    for i in range(0, len(bits), 8):
        val = 0
        for j in range(8):
            val = (val << 1) | int(bits[i + j])
        out.append(val)
    return bytes(out)


def _crc_byte_remainder(data, poly, width):
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    crc = 0
    for byte in data:
        crc ^= byte << (width - 8) if width > 8 else byte
        if width <= 8:
            crc ^= byte
        for _ in range(8):
            if crc & top:
                crc = ((crc << 1) ^ poly) & mask
            else:
                crc = (crc << 1) & mask
    return crc


def _crc8_byte(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        rem = _crc8_byte(_bits_to_bytes(info_bits))
        crc_bits = np.array([(rem >> i) & 1 for i in range(7, -1, -1)], dtype=int)
    else:
        rem = _crc_byte_remainder(_bits_to_bytes(info_bits), CRC16_POLY, 16)
        crc_bits = np.array([(rem >> i) & 1 for i in range(15, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        return _crc8_byte(_bits_to_bytes(bits)) == 0
    return _crc_byte_remainder(_bits_to_bytes(bits), CRC16_POLY, 16) == 0


class _SCLPath:
    __slots__ = ("engine", "pm")

    def __init__(self, engine, pm=0.0):
        self.engine = engine
        self.pm = pm


class SCLDecoder:
    """SCL 译码器（基于 Permuted SCD 内核）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.frozen_set = set(np.where(self.frozen_bits.astype(bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits.astype(bool))[0]

    def _penalty(self, llr, u):
        hard = hard_decision(llr)
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_perm = llr_ch[rev]

        engine0 = _SCDEngine(self.N, self.frozen_set)
        engine0.L[:, 0] = llr_perm
        paths = [_SCLPath(engine0, 0.0)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                eng = path.engine
                eng._update_llrs(l)
                llr_bit = eng.L[l, self.n]
                if l in self.frozen_set:
                    child_eng = copy.deepcopy(eng)
                    child_eng.B[l, self.n] = 0
                    child_eng._update_bits(l)
                    pm = path.pm + self._penalty(llr_bit, 0)
                    candidates.append(_SCLPath(child_eng, pm))
                else:
                    for u in (0, 1):
                        child_eng = copy.deepcopy(eng)
                        child_eng.B[l, self.n] = u
                        child_eng._update_bits(l)
                        pm = path.pm + self._penalty(llr_bit, u)
                        candidates.append(_SCLPath(child_eng, pm))

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        best = paths[0]
        u_hat = best.engine.B[:, self.n].astype(int)

        if self.crc_length > 0:
            for p in paths:
                uh = p.engine.B[:, self.n].astype(int)
                info_bits = uh[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    return uh.copy(), p.pm

        return u_hat.copy(), best.pm
