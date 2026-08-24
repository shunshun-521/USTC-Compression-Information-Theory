"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
    hard_decision,
)
from encoder import bit_reversed


CRC8_POLY_BITS = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
CRC16_POLY_BITS = np.array(
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.int8
)


def _crc_poly_bits(crc_length):
    return CRC8_POLY_BITS if crc_length == 8 else CRC16_POLY_BITS


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly_bits(crc_length)
    n = crc_length
    padded = np.concatenate([info_bits, np.zeros(n, dtype=np.int8)])
    for i in range(len(info_bits)):
        if padded[i]:
            padded[i:i + n + 1] ^= poly
    crc_bits = padded[len(info_bits):len(info_bits) + n]
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly_bits(crc_length)
    n = crc_length
    work = bits.copy()
    for i in range(len(bits) - n):
        if work[i]:
            work[i:i + n + 1] ^= poly
    return np.all(work[-n:] == 0)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = hard_decision(llr)
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [{
            "pm": 0.0,
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
            "u_hat": np.zeros(self.N, dtype=np.int8),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    penalty = self._pm_penalty(llr, 0)
                    child = {
                        "pm": path["pm"] + penalty,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    child["u_hat"][l] = 0
                    child["B"][l, self.n] = 0
                    self._update_bits(child["B"], l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = {
                            "pm": path["pm"] + self._pm_penalty(llr, bit),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        child["u_hat"][l] = bit
                        child["B"][l, self.n] = bit
                        self._update_bits(child["B"], l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u_hat"], self.crc_length)]
            best = min(valid, key=lambda p: p["pm"]) if valid else min(paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
