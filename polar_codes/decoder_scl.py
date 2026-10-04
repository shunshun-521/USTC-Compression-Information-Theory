"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
from copy import deepcopy

import numpy as np

from decoder_sc import sc_decode
from sc_pscd import SCD
from sc_pscd_utils import bit_reversed, hard_decision


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if len(bits) < crc_length:
        return False
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class _PC:
    pass


class _PathState:
    def __init__(self, scd, pm=0.0):
        self.scd = scd
        self.pm = pm


class SCLDecoder:
    """SCL 译码器（基于 PSCD 状态复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _new_scd(self, llr_ch):
        pc = _PC()
        pc.N = self.N
        pc.n = self.n
        pc.frozen = list(self.frozen_set)
        pc.likelihoods = llr_ch
        return SCD(pc)

    def _step_bit(self, state, l):
        scd = state.scd
        scd.update_llrs(l)
        llr = scd.L[l, scd.myPC.n]
        return llr

    def _apply_bit(self, state, l, u_bit):
        scd = state.scd
        if l in self.frozen_set:
            scd.B[l, scd.myPC.n] = 0
        else:
            scd.B[l, scd.myPC.n] = u_bit
        scd.update_bits(l)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [_PathState(self._new_scd(llr_ch), 0.0)]
        decode_order = [bit_reversed(i, self.n) for i in range(self.N)]

        for l in decode_order:
            candidates = []
            for path in paths:
                base = _PathState(deepcopy(path.scd), path.pm)
                llr = self._step_bit(base, l)
                if l in self.frozen_set:
                    candidates.append((base.pm + (abs(llr) if llr < 0 else 0.0), base, 0))
                else:
                    for u in (0, 1):
                        pen = 0.0 if (u == 0 and llr >= 0) or (u == 1 and llr < 0) else abs(llr)
                        candidates.append((base.pm + pen, base, u))

            candidates.sort(key=lambda t: t[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, parent, u in candidates:
                child = _PathState(deepcopy(parent.scd), pm)
                self._apply_bit(child, l, u)
                new_paths.append(child)
            paths = new_paths

        best = None
        if self.crc_length > 0:
            info_mask = self.frozen_bits == 0
            for path in paths:
                u_hat = path.scd.B[:, self.n].astype(int)
                if crc_check(u_hat[info_mask], self.crc_length):
                    if best is None or path.pm < best.pm:
                        best = path
        if best is None:
            best = min(paths, key=lambda p: p.pm)

        u_hat = best.scd.B[:, self.n].astype(int)
        return u_hat, best.pm


def scl_equivalent_sc(llr_ch, frozen_bits):
    u_scl, _ = SCLDecoder(len(llr_ch), frozen_bits, list_size=1, crc_length=0).decode(llr_ch)
    u_sc = sc_decode(llr_ch, frozen_bits)
    return np.array_equal(u_scl, u_sc)
