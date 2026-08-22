"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_utils import (
    active_bit_level,
    active_llr_level,
    hard_decision,
    lower_llr,
    upper_llr,
)
from encoder import bit_reversed


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_step(reg, bit, poly, crc_length):
    reg ^= int(bit) << (crc_length - 1)
    for _ in range(8 if crc_length == 8 else 1):
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg = _crc_step(reg, bit, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for bit in bits:
        reg = _crc_step(reg, bit, poly, crc_length)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 风格路径管理）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _pm_update(self, pm, llr, u):
        expected = 0 if llr >= 0 else 1
        return pm + (0.0 if u == expected else abs(llr))

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [
            {
                "pm": 0.0,
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.full((N, n + 1), np.nan),
                "u": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []
            for pid, path in enumerate(paths):
                self._update_llrs(path["L"], path["B"], l)
                llr = path["L"][l, n]
                if l in self.frozen_set:
                    pm = self._pm_update(path["pm"], llr, 0)
                    candidates.append((pm, pid, 0))
                else:
                    for u in (0, 1):
                        pm = self._pm_update(path["pm"], llr, u)
                        candidates.append((pm, pid, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, pid, u_bit in candidates:
                src = paths[pid]
                L = src["L"].copy()
                B = src["B"].copy()
                u = src["u"].copy()
                u[l] = u_bit
                B[l, n] = 0 if l in self.frozen_set else u_bit
                self._update_bits(B, l)
                new_paths.append({"pm": pm, "L": L, "B": B, "u": u})
            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for i, p in enumerate(paths):
                info_bits = p["u"][self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(i)
            if valid:
                best = min(valid, key=lambda i: paths[i]["pm"])
            else:
                best = int(np.argmin([p["pm"] for p in paths]))
        else:
            best = int(np.argmin([p["pm"] for p in paths]))

        return paths[best]["u"], paths[best]["pm"]
