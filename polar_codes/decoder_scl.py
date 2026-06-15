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
    preprocess_channel_llr,
)


CRC8_POLY_BITS = [1, 0, 0, 0, 0, 0, 1, 1, 1]
CRC16_POLY_BITS = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _poly_bits(crc_length):
    if crc_length == 8:
        return CRC8_POLY_BITS
    if crc_length == 16:
        return CRC16_POLY_BITS
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _poly_bits(crc_length)
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    n_info = len(info_bits)
    for i in range(n_info):
        if msg[i] == 1:
            msg[i : i + len(poly)] ^= poly
    return np.concatenate([info_bits, msg[-crc_length:]])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = _poly_bits(crc_length)
    msg = bits.copy()
    n_info = len(bits) - crc_length
    for i in range(n_info):
        if msg[i] == 1:
            msg[i : i + len(poly)] ^= poly
    return np.all(msg[-crc_length:] == 0)


def _pm_penalty(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


def _update_llrs_path(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        N = L.shape[0]
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = _lower_llr(
                    L[j, s],
                    L[j - branch_size, s],
                    int(B[j - branch_size, s + 1]),
                )


def _update_bits_path(B, l, n):
    if l < (1 << (n - 1)):
        return
    N = B.shape[0]
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, 0] = llr_ch
        return {"pm": 0.0, "L": L, "B": B, "u_hat": np.zeros(self.N, dtype=int)}

    def _clone_path(self, path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "u_hat": path["u_hat"].copy(),
        }

    def decode(self, llr_ch):
        llr_ch = preprocess_channel_llr(llr_ch)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs_path(path["L"], path["B"], l, self.n)
                llr = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    new_path = self._clone_path(path)
                    new_path["u_hat"][l] = 0
                    new_path["B"][l, self.n] = 0
                    new_path["pm"] += _pm_penalty(llr, 0)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = self._clone_path(path)
                        new_path["u_hat"][l] = u
                        new_path["B"][l, self.n] = u
                        new_path["pm"] += _pm_penalty(llr, u)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

            for path in paths:
                _update_bits_path(path["B"], l, self.n)

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
