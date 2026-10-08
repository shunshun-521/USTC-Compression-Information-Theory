"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import copy

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _lower_llr,
    bit_reversed_index,
    f_boxplus_scalar,
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
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(self.frozen_bits == 0)[0]

    def _branch_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_boxplus_scalar(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
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
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        decode_order = [bit_reversed_index(i, n) for i in range(N)]

        for l in decode_order:
            new_paths = []
            for path in paths:
                L, B = path["L"], path["B"]
                self._update_llrs(L, B, l)
                llr_root = L[l, n]

                if l in self.frozen_set:
                    pm = path["pm"] + self._branch_penalty(llr_root, 0)
                    npath = {
                        "pm": pm,
                        "L": L.copy(),
                        "B": copy.deepcopy(B),
                    }
                    npath["B"][l, n] = 0
                    self._update_bits(npath["B"], l)
                    new_paths.append(npath)
                else:
                    for u in (0, 1):
                        pm = path["pm"] + self._branch_penalty(llr_root, u)
                        npath = {
                            "pm": pm,
                            "L": L.copy(),
                            "B": copy.deepcopy(B),
                        }
                        npath["B"][l, n] = u
                        self._update_bits(npath["B"], l)
                        new_paths.append(npath)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        candidates = paths
        if self.crc_length > 0:
            valid = []
            for p in candidates:
                u_hat = p["B"][:, n].astype(int)
                payload = u_hat[self.info_positions]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                candidates = valid

        best = min(candidates, key=lambda p: p["pm"])
        u_hat = best["B"][:, n].astype(int)
        return u_hat, best["pm"]
