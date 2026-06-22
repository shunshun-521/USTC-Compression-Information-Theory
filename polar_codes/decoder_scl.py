"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _SCD, bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_register(bits, degree, poly):
    crc = np.zeros(degree, dtype=int)
    for bit in bits:
        feedback = int(bit) ^ crc[0]
        crc[:-1] = crc[1:]
        crc[-1] = feedback
        for i in range(degree):
            if (poly >> (degree - 1 - i)) & 1:
                crc[i] ^= feedback
    return crc


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_register(info_bits, crc_length, poly)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return np.all(_crc_register(bits, crc_length, poly) == 0)


def _path_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if hard == bit else abs(llr)


class _SCLPath:
    __slots__ = ("pm", "scd", "u_hat", "stage")

    def __init__(self, llr_ch, frozen_set, n):
        self.pm = 0.0
        self.scd = _SCD(llr_ch, frozen_set, n)
        self.u_hat = np.zeros(len(llr_ch), dtype=int)
        self.stage = 0


class SCLDecoder:
    """SCL 译码器（基于 Permuted SCD 路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def _advance_path(self, path, l, bit):
        path.scd._update_llrs(l)
        llr0 = path.scd.L[l, self.n]
        path.pm += _path_penalty(llr0, bit)
        path.scd.B[l, self.n] = bit
        path.u_hat[l] = bit
        path.scd._update_bits(l)
        path.stage += 1

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_SCLPath(llr_ch, self.frozen_set, self.n)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                if l in self.frozen_set:
                    child = _SCLPath(llr_ch, self.frozen_set, self.n)
                    child.pm = path.pm
                    child.scd.L = path.scd.L.copy()
                    child.scd.B = path.scd.B.copy()
                    child.u_hat = path.u_hat.copy()
                    child.stage = path.stage
                    self._advance_path(child, l, 0)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = _SCLPath(llr_ch, self.frozen_set, self.n)
                        child.pm = path.pm
                        child.scd.L = path.scd.L.copy()
                        child.scd.B = path.scd.B.copy()
                        child.u_hat = path.u_hat.copy()
                        child.stage = path.stage
                        self._advance_path(child, l, bit)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
