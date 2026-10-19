"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 PSC 树更新
"""
import math
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr_min_sum,
    _upper_llr_min_sum,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_digest(info_bits, crc_length):
    """基于 CRC-8/16 多项式的 LFSR _digest（自洽校验，用于 CA-SCL）。"""
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    digest = _crc_digest(info_bits, crc_length)
    crc_bits = np.array(
        [(digest >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    rx = bits[-crc_length:]
    digest = _crc_digest(info, crc_length)
    expected = np.array(
        [(digest >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.array_equal(rx, expected)


class SCLDecoder:
    """SCL 译码器（每条路径维护 PSC 的 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs(self, L, B, l):
        start_s = self.n - _active_llr_level(l, self.n)
        for s in range(start_s, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_min_sum(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr_min_sum(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        end_s = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, end_s, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        L = int(self.list_size)

        paths = []
        L0 = np.full((N, n + 1), np.nan, dtype=np.float64)
        L0[:, 0] = llr_ch
        paths.append(
            {
                "pm": 0.0,
                "L": L0,
                "B": np.zeros((N, n + 1), dtype=int),
                "u_hat": np.zeros(N, dtype=int),
            }
        )

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []
            for pidx, path in enumerate(paths):
                self._update_llrs(path["L"], path["B"], l)
                llr_bit = path["L"][l, n]
                if self.frozen_bits[l]:
                    pen = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    candidates.append((path["pm"] + pen, pidx, 0))
                else:
                    pen0 = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    pen1 = abs(llr_bit) if llr_bit >= 0 else 0.0
                    candidates.append((path["pm"] + pen0, pidx, 0))
                    candidates.append((path["pm"] + pen1, pidx, 1))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:L]

            new_paths = []
            for pm, parent, bit in candidates:
                src = paths[parent]
                child = {
                    "pm": pm,
                    "L": src["L"].copy(),
                    "B": src["B"].copy(),
                    "u_hat": src["u_hat"].copy(),
                }
                child["B"][l, n] = bit
                child["u_hat"][l] = bit
                self._update_bits(child["B"], l)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = [
                (p["pm"], idx)
                for idx, p in enumerate(paths)
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda x: x[0])[1]
            else:
                best = int(np.argmin([p["pm"] for p in paths]))
        else:
            best = int(np.argmin([p["pm"] for p in paths]))

        best_path = paths[best]
        return best_path["u_hat"].copy(), best_path["pm"]
