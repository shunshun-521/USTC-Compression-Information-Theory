"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _pm_update,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], int(top_bit))

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] ^ B[j - branch_size, s]) & 1
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L0 = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B0 = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L0[:, 0] = llr_ch

        paths = [{"pm": 0.0, "L": L0.copy(), "B": B0.copy()}]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                L = path["L"]
                B = path["B"]
                self._update_llrs(L, B, l)
                llr_bit = L[l, self.n]

                if self.frozen_bits[l]:
                    pm = _pm_update(path["pm"], llr_bit, 0)
                    B[l, self.n] = 0
                    self._update_bits(B, l)
                    new_paths.append({"pm": pm, "L": L, "B": B})
                else:
                    for u in (0, 1):
                        Lc = L.copy()
                        Bc = B.copy()
                        pm = _pm_update(path["pm"], llr_bit, u)
                        Bc[l, self.n] = u
                        self._update_bits(Bc, l)
                        new_paths.append({"pm": pm, "L": Lc, "B": Bc})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                payload = p["B"][:, self.n][self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        u_hat = best["B"][:, self.n].astype(int)
        return u_hat, best["pm"]
