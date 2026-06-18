"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于置换 SC 树结构
"""
import math
import sys
from pathlib import Path

import numpy as np

_REF_DIR = Path(__file__).resolve().parent / "ref_test"
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))

from SCD import SCD  # noqa: E402
from decoder_sc import path_metric_penalty, sc_decode
from encoder import bit_reversal_permutation
from polar_utils import bit_reversed

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_polynomial(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length <= 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _PC:
    __slots__ = ("N", "n", "frozen", "likelihoods")

    def __init__(self, N, n, frozen, likelihoods):
        self.N = N
        self.n = n
        self.frozen = frozen
        self.likelihoods = likelihoods


class _SCLPath:
    """单条 SCL 路径，复用 SCD 的 L/B 矩阵"""

    def __init__(self, N, n, frozen, llr):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.phase = 0
        self.pc = _PC(N, n, frozen, llr)
        self.scd = SCD(self.pc)
        self.frozen_set = set(frozen)

    def copy(self):
        new = _SCLPath.__new__(_SCLPath)
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        new.phase = self.phase
        new.frozen_set = self.frozen_set
        new.pc = _PC(self.pc.N, self.pc.n, self.pc.frozen, self.pc.likelihoods.copy())
        new.scd = SCD(new.pc)
        new.scd.L = self.scd.L.copy()
        new.scd.B = self.scd.B.copy()
        return new

    def current_llr(self):
        l = bit_reversed(self.phase, self.pc.n)
        self.scd.update_llrs(l)
        return self.scd.L[l, self.pc.n]

    def decide(self, u_bit):
        l = bit_reversed(self.phase, self.pc.n)
        self.scd.B[l, self.pc.n] = u_bit
        self.u_hat[l] = u_bit
        self.scd.update_bits(l)
        self.phase += 1


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen = np.where(self.frozen_bits)[0]
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr = llr_ch[self.br].astype(np.float64)
        paths = [_SCLPath(self.N, self.n, self.frozen, llr)]

        for _ in range(self.N):
            candidates = []
            for path in paths:
                llr_bit = path.current_llr()
                l = bit_reversed(path.phase, self.n)
                if l in path.frozen_set:
                    new_path = path.copy()
                    new_path.pm += path_metric_penalty(llr_bit, 0)
                    new_path.decide(0)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = path.copy()
                        new_path.pm += path_metric_penalty(llr_bit, u_bit)
                        new_path.decide(u_bit)
                        candidates.append(new_path)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid or paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
