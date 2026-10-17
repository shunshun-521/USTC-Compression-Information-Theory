"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_scalar,
    _lower_llr,
    _upper_llr,
    sc_decode,
)
from encoder import bit_reversal_permutation
from channel import reorder_llr_for_decoder

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_bits(info_bits, crc_length):
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    width = crc_length
    top = 1 << (width - 1)
    mask = (1 << width) - 1
    poly_shifted = poly if crc_length == 16 else (poly << (width - 8))
    for b in np.asarray(info_bits, dtype=np.int8):
        reg ^= int(b) << (width - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly_shifted) & mask
            else:
                reg = (reg << 1) & mask
    return np.array([(reg >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    return np.concatenate([info_bits, _crc_bits(info_bits, crc_length)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class SCLDecoder:
    """SCL 译码器（分层 L/B，路径列表裁剪）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = reorder_llr_for_decoder(llr_ch, br)

        N, n = self.N, self.n
        paths = [{"pm": 0.0, "L": np.zeros((N, n + 1), dtype=np.float64), "B": np.zeros((N, n + 1), dtype=np.int8)}]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed_scalar(i, n)
            new_paths = []

            for path in paths:
                L, B = path["L"], path["B"]
                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                        else:
                            top_bit = B[j - branch_size, s + 1]
                            L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

                llr0 = L[l, n]
                if self.frozen_bits[l]:
                    pen = abs(llr0) if llr0 < 0 else 0.0
                    child = {
                        "pm": path["pm"] + pen,
                        "L": L.copy(),
                        "B": B.copy(),
                    }
                    child["B"][l, n] = 0
                    self._update_bits(child["B"], l, n)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        pen = 0.0
                        if (llr0 >= 0 and u == 1) or (llr0 < 0 and u == 0):
                            pen = abs(llr0)
                        child = {
                            "pm": path["pm"] + pen,
                            "L": L.copy(),
                            "B": B.copy(),
                        }
                        child["B"][l, n] = u
                        self._update_bits(child["B"], l, n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        u_hat = np.zeros(N, dtype=np.int8)
        for l in range(N):
            u_hat[l] = paths[0]["B"][l, n]

        crc_pass = []
        for path in paths:
            u = np.array([path["B"][l, n] for l in range(N)], dtype=np.int8)
            if self.crc_length > 0:
                info_bits = u[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append((path["pm"], u))
        if self.crc_length > 0 and crc_pass:
            crc_pass.sort(key=lambda x: x[0])
            best_u, best_pm = crc_pass[0][1], crc_pass[0][0]
        else:
            best = min(paths, key=lambda p: p["pm"])
            best_u = np.array([best["B"][l, n] for l in range(N)], dtype=np.int8)
            best_pm = best["pm"]

        return best_u, best_pm

    @staticmethod
    def _update_bits(B, l, n):
        if l < B.shape[0] // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            N = B.shape[0]
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]
