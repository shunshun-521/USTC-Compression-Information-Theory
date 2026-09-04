"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
)
from encoder import bit_reversed


CRC_POLYNOMIALS = {
    8: 0x107,    # x^8 + x^2 + x + 1 (含隐式最高位)
    16: 0x110021,  # CRC-16-IBM (含隐式最高位)
}


def _crc_remainder(bits, crc_length):
    """CRC 余数。"""
    poly = CRC_POLYNOMIALS[crc_length]
    crc = 0
    for bit in bits:
        crc ^= int(bit) << crc_length
        for _ in range(8 if crc_length == 8 else 16):
            if crc & (1 << crc_length):
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
            crc &= (1 << crc_length) - 1
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 crc_length 位是否为正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    return _crc_remainder(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

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
        return L[l, self.n]

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

    @staticmethod
    def _pm_update(pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [{
            "pm": 0.0,
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for l in self.decode_order:
            candidates = []
            for path in paths:
                L = path["L"]
                B = path["B"]
                llr = self._update_llrs(L, B, l)

                if l in self.frozen_set:
                    pm = self._pm_update(path["pm"], llr, 0)
                    B[l, self.n] = 0
                    self._update_bits(B, l)
                    candidates.append({"pm": pm, "L": L, "B": B})
                else:
                    for bit in (0, 1):
                        Lc = path["L"].copy()
                        Bc = path["B"].copy()
                        pm = self._pm_update(path["pm"], llr, bit)
                        Bc[l, self.n] = bit
                        self._update_bits(Bc, l)
                        candidates.append({"pm": pm, "L": Lc, "B": Bc})

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best = min(paths, key=lambda p: p["pm"])
        u_hat = best["B"][:, self.n].copy()

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["B"][:, self.n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p["pm"])
                u_hat = best["B"][:, self.n].copy()

        return u_hat, best["pm"]
