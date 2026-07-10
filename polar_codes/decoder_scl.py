"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    bit_reversed_index,
    active_llr_level,
    active_bit_level,
)


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    poly = CRC_POLYNOMIALS[crc_length]
    info_bits = np.asarray(info_bits, dtype=np.int8)
    msg = list(map(int, info_bits)) + [0] * crc_length
    n = len(msg)
    for i in range(n - crc_length):
        if msg[i] == 1:
            for j in range(crc_length + 1):
                if (poly >> j) & 1:
                    msg[i + j] ^= 1
    crc_bits = np.array(msg[-crc_length:], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    poly = CRC_POLYNOMIALS[crc_length]
    msg = list(map(int, np.asarray(bits, dtype=np.int8)))
    n = len(msg)
    for i in range(n - crc_length):
        if msg[i] == 1:
            for j in range(crc_length + 1):
                if (poly >> j) & 1:
                    msg[i + j] ^= 1
    return all(x == 0 for x in msg[-crc_length:])


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    @staticmethod
    def _penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = []
        L0 = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B0 = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L0[:, 0] = llr_ch
        paths.append({"pm": 0.0, "L": L0, "B": B0})

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            new_paths = []
            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr0 = path["L"][l, self.n]
                if self.frozen_bits[l]:
                    pm = path["pm"] + self._penalty(llr0, 0)
                    B = path["B"].copy()
                    B[l, self.n] = 0
                    self._update_bits(B, l)
                    new_paths.append({"pm": pm, "L": path["L"], "B": B})
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + self._penalty(llr0, bit)
                        L = path["L"].copy()
                        B = path["B"].copy()
                        B[l, self.n] = bit
                        self._update_bits(B, l)
                        new_paths.append({"pm": pm, "L": L, "B": B})
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_hat = p["B"][:, self.n]
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["B"][:, self.n].astype(int), best["pm"]
