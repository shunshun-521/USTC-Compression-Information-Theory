"""
极化码 SCL（串行抵消列表）译码器
基于 Permuted SCD，支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np

from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
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
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def _path_llr(path, l, n, N):
    L, B = path["L"], path["B"]
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = _lower_llr(
                    L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                )
    return L[l, n]


def _path_update_bits(path, l, n, N):
    if l < N / 2:
        return
    L, B = path["L"], path["B"]
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_mask = self.frozen_bits == 0

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_ch
        return {"L": L, "B": B, "pm": 0.0}

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        paths = [self._new_path(llr_ch)]

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []
            for path in paths:
                llr = _path_llr(path, l, n, N)
                if l in self.frozen_set:
                    path["B"][l, n] = 0
                    pen = 0.0 if llr >= 0 else abs(llr)
                    path["pm"] += pen
                    _path_update_bits(path, l, n, N)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = copy.deepcopy(path)
                        pen = 0.0 if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0) else abs(llr)
                        child["pm"] = path["pm"] + pen
                        child["B"][l, n] = bit
                        _path_update_bits(child, l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = min(paths, key=lambda p: p["pm"])
        u_hat = np.nan_to_num(best["B"][:, n], nan=0.0).astype(int)

        if self.crc_length > 0:
            info_bits = u_hat[self.info_mask]
            passing = []
            for p in paths:
                ub = np.nan_to_num(p["B"][:, n], nan=0.0).astype(int)
                if crc_check(ub[self.info_mask], self.crc_length):
                    passing.append((p["pm"], ub))
            if passing:
                passing.sort(key=lambda x: x[0])
                u_hat = passing[0][1]

        return u_hat, best["pm"]
