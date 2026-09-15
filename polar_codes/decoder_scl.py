"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 PSC 架构
"""
import math
import numpy as np
from decoder_sc import (
    _SCDState,
    _prepare_channel_llr,
    bit_reversed_index,
    hard_decision,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    remainder = _crc_remainder(padded, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("state", "pm")

    def __init__(self, N, frozen_bits):
        self.state = _SCDState(N, frozen_bits)
        self.pm = 0.0


class SCLDecoder:
    """SCL 译码器（PSC + 路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _new_path(self, llr_ch):
        path = _Path(self.N, self.frozen_bits)
        path.state.L[:, 0] = _prepare_channel_llr(llr_ch)
        return path

    @staticmethod
    def _pm_update(pm, llr, u_bit):
        hard = hard_decision(llr)
        if u_bit != hard:
            pm += abs(llr)
        return pm

    def _clone_path(self, path):
        new_path = _Path(self.N, self.frozen_bits)
        new_path.state.L = path.state.L.copy()
        new_path.state.B = path.state.B.copy()
        new_path.pm = path.pm
        return new_path

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                path.state.update_llrs(l)
                llr = path.state.L[l, self.n]

                if l in self.frozen_set:
                    pm = self._pm_update(path.pm, llr, 0)
                    new_path = self._clone_path(path)
                    new_path.pm = pm
                    new_path.state.B[l, self.n] = 0
                    new_path.state.update_bits(l)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        pm = self._pm_update(path.pm, llr, u_bit)
                        new_path = self._clone_path(path)
                        new_path.pm = pm
                        new_path.state.B[l, self.n] = u_bit
                        new_path.state.update_bits(l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = [
                p for p in paths
                if crc_check(p.state.B[:, self.n][info_idx], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.state.B[:, self.n].astype(np.int8), best.pm
