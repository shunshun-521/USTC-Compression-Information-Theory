"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    precompute_sc_indices,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


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
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（非递归 LLR 树，与 SC 相同更新规则）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, 0] = llr_ch
        return {"pm": 0.0, "L": L, "B": B, "u_hat": np.zeros(self.N, dtype=np.int8)}

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        L, B = path["L"], path["B"]
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    @staticmethod
    def _branch_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for l in self.decode_order:
            expanded = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path["L"][l, self.n]
                if self.frozen_bits[l]:
                    child = copy.deepcopy(path)
                    child["pm"] += self._branch_penalty(llr, 0)
                    child["B"][l, self.n] = 0
                    child["u_hat"][l] = 0
                    self._update_bits(child, l)
                    expanded.append(child)
                else:
                    for u in (0, 1):
                        child = copy.deepcopy(path)
                        child["pm"] += self._branch_penalty(llr, u)
                        child["B"][l, self.n] = u
                        child["u_hat"][l] = u
                        self._update_bits(child, l)
                        expanded.append(child)

            expanded.sort(key=lambda p: p["pm"])
            paths = expanded[: self.list_size]

        if self.crc_length > 0:
            info_pos = np.where(~self.frozen_bits)[0]
            for p in paths:
                if crc_check(p["u_hat"][info_pos], self.crc_length):
                    return p["u_hat"].copy(), p["pm"]
        best = paths[0]
        return best["u_hat"].copy(), best["pm"]
