"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.uint8)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length))


class SCLDecoder:
    """SCL 译码器（路径列表 + PM）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L_size = list_size
        self.crc_length = crc_length

    def _init_paths(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, 0] = llr_ch
        return [
            {"L": L, "B": B, "pm": 0.0, "u": np.zeros(self.N, dtype=int)}
            for _ in range(1)
        ]

    def _update_llrs(self, path, l):
        L = path["L"]
        B = path["B"]
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        B = path["B"]
        n = self.n
        N = self.N
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._init_paths(llr_ch)

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path["L"][l, self.n]
                if np.isnan(llr_bit):
                    llr_bit = 0.0

                if self.frozen_bits[l]:
                    u = 0
                    pm = path["pm"] + (abs(llr_bit) if llr_bit < 0 else 0.0)
                    p2 = dict(path)
                    p2["pm"] = pm
                    p2["L"] = path["L"].copy()
                    p2["B"] = path["B"].copy()
                    p2["u"] = path["u"].copy()
                    p2["u"][l] = u
                    p2["B"][l, self.n] = u
                    self._update_bits(p2, l)
                    new_paths.append(p2)
                else:
                    for u in (0, 1):
                        pm = path["pm"]
                        hard = 0 if llr_bit >= 0 else 1
                        if u != hard:
                            pm += abs(llr_bit)
                        p2 = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": pm,
                            "u": path["u"].copy(),
                        }
                        p2["u"][l] = u
                        p2["B"][l, self.n] = u
                        self._update_bits(p2, l)
                        new_paths.append(p2)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.L_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]
