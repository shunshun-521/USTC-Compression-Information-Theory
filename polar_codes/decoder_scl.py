"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    lower_llr,
    upper_llr,
)
from encoder import bit_reversed


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0xE0  # CRC-8 (0x07) 的反射多项式
    if crc_length == 16:
        return 0xA001  # CRC-16 (0x8005) 的反射多项式
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & msb:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    reg = _crc_remainder(padded, crc_length)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, crc_bits]).astype(np.int8)


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确。"""
    return _crc_remainder(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Permuted SCD + Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def _branch_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def _clone_path(self, path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "u_hat": path["u_hat"].copy(),
        }

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s],
                        int(B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        L, B = path["L"], path["B"]
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{
            "pm": 0.0,
            "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "B": np.full((self.N, self.n + 1), np.nan),
            "u_hat": np.zeros(self.N, dtype=np.int8),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for l in self.decode_order:
            for path in paths:
                self._update_llrs(path, l)

            new_paths = []
            for path in paths:
                llr = path["L"][l, self.n]
                if l in self.frozen_set:
                    child = self._clone_path(path)
                    child["pm"] += self._branch_penalty(llr, 0)
                    child["B"][l, self.n] = 0
                    child["u_hat"][l] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = self._clone_path(path)
                        child["pm"] += self._branch_penalty(llr, bit)
                        child["B"][l, self.n] = bit
                        child["u_hat"][l] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:self.list_size]

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u_hat"], self.crc_length)]
            best = valid[0] if valid else paths[0]
        else:
            best = paths[0]

        return best["u_hat"].astype(int), best["pm"]
