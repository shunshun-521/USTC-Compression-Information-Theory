"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    _lower_llr,
    _upper_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_polynomial(crc_length):
    return CRC8_POLY if crc_length == 8 else CRC16_POLY


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_polynomial(crc_length)
    mask = (1 << crc_length) - 1

    reg = 0
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    for bit in padded:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb ^ int(bit):
            reg ^= poly & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否满足 CRC 约束。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


def _llr_to_bit(llr):
    return 0 if llr >= 0 else 1


def _path_metric_update(pm, llr, u):
    expected = _llr_to_bit(llr)
    if u != expected:
        return pm + abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _init_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch
        return {"L": L, "B": B, "pm": 0.0, "u_hat": np.full(self.N, -1, dtype=int)}

    def _advance_llr(self, path, phi):
        L, B = path["L"], path["B"]
        l = bit_reversed(phi, self.n)

        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )
        return l, L[l, self.n]

    def _update_bits(self, path, l, u_val):
        B = path["B"]
        B[l, self.n] = u_val
        path["u_hat"][l] = u_val

        if l < self.N // 2:
            return

        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        paths = [self._init_path(llr_ch[rev])]

        for phi in range(self.N):
            new_paths = []
            l_idx = bit_reversed(phi, self.n)

            if l_idx in self.frozen_set:
                for path in paths:
                    p = copy.deepcopy(path)
                    _, cur_llr = self._advance_llr(p, phi)
                    p["pm"] = _path_metric_update(p["pm"], cur_llr, 0)
                    self._update_bits(p, l_idx, 0)
                    new_paths.append(p)
            else:
                for path in paths:
                    p_base = copy.deepcopy(path)
                    _, cur_llr = self._advance_llr(p_base, phi)
                    for u_val in (0, 1):
                        p = copy.deepcopy(p_base)
                        p["pm"] = _path_metric_update(p["pm"], cur_llr, u_val)
                        self._update_bits(p, l_idx, u_val)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best_crc = None
        if self.crc_length > 0:
            for path in paths:
                u = path["u_hat"].copy()
                u[u < 0] = 0
                if crc_check(u, self.crc_length):
                    if best_crc is None or path["pm"] < best_crc["pm"]:
                        best_crc = path

        best_path = best_crc if best_crc is not None else min(paths, key=lambda p: p["pm"])
        u_hat = best_path["u_hat"].copy()
        u_hat[u_hat < 0] = 0
        return u_hat, best_path["pm"]
