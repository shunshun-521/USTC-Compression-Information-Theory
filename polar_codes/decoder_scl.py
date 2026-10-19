"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    bit_reversed,
    active_llr_level,
    active_bit_level,
    _hard_decision,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_bits(info_bits, crc_length):
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in np.asarray(info_bits, dtype=np.int8).ravel():
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)]


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    crc = np.array(_crc_bits(info_bits, crc_length), dtype=np.int8)
    return np.concatenate([info_bits, crc])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8).ravel()
    return _crc_bits(bits, crc_length) == [0] * crc_length


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, 0] = llr_ch
        return {"L": L, "B": B, "pm": 0.0, "u_hat": np.zeros(self.N, dtype=np.int8)}

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        L, B = path["L"], path["B"]
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr_val, u_bit):
        hard = _hard_decision(llr_val)
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        rev = np.array([bit_reversed(i, self.n) for i in range(self.N)], dtype=np.int64)
        llr_ordered = np.asarray(llr_ch, dtype=np.float64)[rev]

        paths = [self._new_path(llr_ordered)]
        phase_order = [bit_reversed(i, self.n) for i in range(self.N)]

        for l in phase_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr0 = path["L"][l, self.n]
                if self.frozen_bits[l]:
                    pm = path["pm"] + self._pm_penalty(llr0, 0)
                    candidates.append((pm, path, 0))
                else:
                    for u in (0, 1):
                        pm = path["pm"] + self._pm_penalty(llr0, u)
                        candidates.append((pm, path, u))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, path, u in candidates[: self.list_size]:
                npth = {
                    "L": path["L"].copy(),
                    "B": path["B"].copy(),
                    "pm": pm,
                    "u_hat": path["u_hat"].copy(),
                }
                npth["B"][l, self.n] = 0 if self.frozen_bits[l] else u
                npth["u_hat"][l] = npth["B"][l, self.n]
                self._update_bits(npth, l)
                new_paths.append(npth)
            paths = new_paths

        best = min(paths, key=lambda p: p["pm"])
        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p["pm"])
        return best["u_hat"], best["pm"]
