"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
    f_operation,
    g_operation,
)


def _crc_bits(info_bits, crc_length):
    if crc_length == 8:
        poly = 0x07
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        return np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    if crc_length == 16:
        poly = 0x8005
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 15
            for _ in range(8):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        return np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    return np.concatenate([info_bits, _crc_bits(info_bits, crc_length)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits[-crc_length:], _crc_bits(bits[:-crc_length], crc_length))


class SCLDecoder:
    """SCL 译码器（Permuted SC + 路径度量）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _new_path(self, llr_br):
        return {
            "pm": 0.0,
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
            "u_hat": np.zeros(self.N, dtype=np.int8),
            "llr_br": llr_br,
        }

    def _init_path(self, path):
        path["L"][:, 0] = path["llr_br"]

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        B = path["B"]
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_br = llr_ch[self.br]

        paths = [self._new_path(llr_br)]
        self._init_path(paths[0])

        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            expanded = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    new_path = copy.deepcopy(path)
                    new_path["pm"] += self._penalty(llr, 0)
                    new_path["B"][l, self.n] = 0
                    new_path["u_hat"][l] = 0
                    self._update_bits(new_path, l)
                    expanded.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = copy.deepcopy(path)
                        new_path["pm"] += self._penalty(llr, u)
                        new_path["B"][l, self.n] = u
                        new_path["u_hat"][l] = u
                        self._update_bits(new_path, l)
                        expanded.append(new_path)

            expanded.sort(key=lambda p: p["pm"])
            paths = expanded[: self.list_size]

        best = min(paths, key=lambda p: p["pm"])
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][info_idx], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p["pm"])

        return best["u_hat"].astype(int), float(best["pm"])
