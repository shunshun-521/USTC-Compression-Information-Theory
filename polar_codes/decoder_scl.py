"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _boxplus,
    _lower_llr,
    precompute_sc_indices,
)


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _crc_polynomial(crc_length)
    reg = 0
    steps = crc_length
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(steps):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.phase_order, _, _ = precompute_sc_indices(N)
        self.info_positions = np.where(self.frozen_bits == 0)[0]

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch
        return {
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=int),
            "L": L,
            "B": B,
        }

    def _copy_path(self, path):
        return {
            "pm": path["pm"],
            "u_hat": path["u_hat"].copy(),
            "L": path["L"].copy(),
            "B": path["B"].copy(),
        }

    def _update_llrs(self, path, l):
        L = path["L"]
        B = path["B"]
        n = self.n
        N = self.N
        start_s = n - _active_llr_level(l, n)
        for s in range(start_s, n):
            block = 1 << (s + 1)
            branch = block >> 1
            for j in range(l, N, block):
                if j % block < branch:
                    L[j, s + 1] = _boxplus(L[j, s], L[j + branch, s])
                else:
                    top_bit = B[j - branch, s + 1]
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        B = path["B"]
        n = self.n
        start_s = n - _active_bit_level(l, n)
        for s in range(n, start_s, -1):
            block = 1 << s
            branch = block >> 1
            for j in range(l, -1, -block):
                if j % block >= branch:
                    B[j - branch, s - 1] = B[j, s] ^ B[j - branch, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for l in self.phase_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    penalty = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    path["pm"] += penalty
                    path["u_hat"][l] = 0
                    path["B"][l, self.n] = 0
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        penalty = 0.0 if (llr_bit >= 0) == (bit == 0) else abs(llr_bit)
                        new_path["pm"] += penalty
                        new_path["u_hat"][l] = bit
                        new_path["B"][l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p["pm"])

        if self.crc_length > 0:
            for path in paths:
                info_bits = path["u_hat"][self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    return path["u_hat"].copy(), path["pm"]

        best = paths[0]
        return best["u_hat"].copy(), best["pm"]
