"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
)
from encoder import bit_reversed


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def _crc_remainder(bits, crc_length, poly):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in np.asarray(bits, dtype=int):
        feedback = ((reg >> (crc_length - 1)) ^ bit) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    remainder = _crc_remainder(info_bits, crc_length, poly)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly(crc_length)
    return _crc_remainder(bits, crc_length, poly) == 0


class SCLDecoder:
    """SCL 译码器（mcba1n 风格 SC 内核）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _new_path(self, llr_ch):
        return {
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=int),
            "u_hat": np.zeros(self.N, dtype=int),
            "pm": 0.0,
        }

    def _update_llrs(self, path, l):
        n = self.n
        N = self.N
        L = path["L"]
        B = path["B"]

        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
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

        n = self.n
        B = path["B"]
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数。返回：u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path["L"][l, self.n]

                if l in self.frozen_set:
                    new_path = copy.deepcopy(path)
                    new_path["pm"] += self._pm_penalty(llr, 0)
                    new_path["B"][l, self.n] = 0
                    new_path["u_hat"][l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = copy.deepcopy(path)
                        new_path["pm"] += self._pm_penalty(llr, bit)
                        new_path["B"][l, self.n] = bit
                        new_path["u_hat"][l] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[:self.list_size]

        pool = paths
        if self.crc_length > 0:
            crc_paths = []
            for path in paths:
                info_bits = path["u_hat"][~self.frozen_bits.astype(bool)]
                if crc_check(info_bits, self.crc_length):
                    crc_paths.append(path)
            if crc_paths:
                pool = crc_paths

        best = min(pool, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
