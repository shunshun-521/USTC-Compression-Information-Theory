"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _hard_decision,
    _lower_llr,
    _upper_llr,
    sc_decode,
)

_CRC_POLY = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC_POLY[crc_length]
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N / 2:
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
        paths = [
            {
                "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
                "B": np.full((self.N, self.n + 1), np.nan),
                "pm": 0.0,
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            new_paths = []
            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr = path["L"][l, self.n]
                if l in self.frozen_set:
                    pen = 0.0 if llr >= 0 else abs(llr)
                    p2 = copy.deepcopy(path)
                    p2["pm"] += pen
                    p2["B"][l, self.n] = 0
                    self._update_bits(p2["B"], l)
                    new_paths.append(p2)
                else:
                    for u_bit in (0, 1):
                        pen = 0.0 if (u_bit == 0 and llr >= 0) or (u_bit == 1 and llr < 0) else abs(llr)
                        p2 = copy.deepcopy(path)
                        p2["pm"] += pen
                        p2["B"][l, self.n] = u_bit
                        self._update_bits(p2["B"], l)
                        new_paths.append(p2)
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = []
            for p in paths:
                bits = p["B"][:, self.n].astype(int)
                if crc_check(bits[info_idx], self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = paths[0]
        return best["B"][:, self.n].astype(int), best["pm"]
