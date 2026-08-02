"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _upper_llr_exact,
    g_operation,
    precompute_sc_indices,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """CRC-8/16 余数（Dallas/Maxim 风格，MSB 先）"""
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    remainder = _crc_remainder(padded, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    info = bits[:-crc_length]
    crc_bits = bits[-crc_length:]
    expected = _crc_remainder(
        np.concatenate([info, np.zeros(crc_length, dtype=np.int8)]), poly, crc_length
    )
    received = sum(int(crc_bits[i]) << (crc_length - 1 - i) for i in range(crc_length))
    return expected == received


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _init_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, 0] = llr_ch
        return {"pm": 0.0, "L": L, "B": B, "u_hat": np.zeros(self.N, dtype=np.int8)}

    def _copy_path(self, path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "u_hat": path["u_hat"].copy(),
        }

    def _update_llr(self, path, l):
        L, B = path["L"], path["B"]
        start_s = self.n - _active_llr_level(l, self.n)
        for s in range(start_s, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_exact(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    bottom_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    if top_bit == 0:
                        L[j, s + 1] = bottom_llr + top_llr
                    else:
                        L[j, s + 1] = bottom_llr - top_llr

    def _update_bits(self, path, l):
        B = path["B"]
        if l < self.N // 2:
            return
        end_s = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, end_s, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _penalty(self, llr_val, u):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._init_path(llr_ch)]

        for l in self.decode_order:
            candidates = []
            for p_idx, path in enumerate(paths):
                self._update_llr(path, l)
                llr_val = path["L"][l, self.n]

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    new_path["pm"] += self._penalty(llr_val, 0)
                    new_path["u_hat"][l] = 0
                    new_path["B"][l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append((new_path["pm"], new_path))
                else:
                    for u in (0, 1):
                        new_path = self._copy_path(path)
                        new_path["pm"] += self._penalty(llr_val, u)
                        new_path["u_hat"][l] = u
                        new_path["B"][l, self.n] = u
                        self._update_bits(new_path, l)
                        candidates.append((new_path["pm"], new_path))

            candidates.sort(key=lambda x: x[0])
            paths = [c[1] for c in candidates[: self.list_size]]

        if self.crc_length > 0:
            valid = [
                p for p in paths if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].astype(int), best["pm"]
