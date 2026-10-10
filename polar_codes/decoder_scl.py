"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np

from encoder import bit_reversed
from scd_ref import SCD

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_bits(data, poly, r):
    reg = 0
    for bit in data:
        reg ^= int(bit) << (r - 1)
        for _ in range(8):
            if reg & (1 << (r - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << r) - 1)
            else:
                reg = (reg << 1) & ((1 << r) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = _crc_bits(info_bits, poly, crc_length)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_bits(bits, poly, crc_length) == 0


def _make_pc(N, n, frozen_set, llr_ch):
    class _PC:
        pass

    pc = _PC()
    pc.N = N
    pc.n = n
    pc.frozen = frozen_set
    pc.likelihoods = llr_ch.copy()
    return pc


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_add(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"scd": SCD(_make_pc(self.N, self.n, self.frozen_set, llr_ch)), "pm": 0.0}]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                scd = path["scd"]
                scd.update_llrs(l)
                llr = scd.L[l, self.n]
                if l in self.frozen_set:
                    scd.B[l, self.n] = 0
                    scd.update_bits(l)
                    path["pm"] += self._pm_add(llr, 0)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        p2 = copy.deepcopy(path)
                        s2 = p2["scd"]
                        s2.B[l, self.n] = u
                        s2.update_bits(l)
                        p2["pm"] += self._pm_add(llr, u)
                        candidates.append(p2)
            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best_any = paths[0]
        best_crc = None
        best_pm = float("inf")
        for p in paths:
            u_hat = p["scd"].B[:, self.n].astype(int)
            if p["pm"] < best_any["pm"]:
                best_any = p
            if self.crc_length > 0:
                if crc_check(u_hat[self.info_positions], self.crc_length) and p["pm"] < best_pm:
                    best_pm = p["pm"]
                    best_crc = u_hat
        if best_crc is not None:
            return best_crc.copy(), best_pm
        u_hat = best_any["scd"].B[:, self.n].astype(int)
        return u_hat.copy(), best_any["pm"]
