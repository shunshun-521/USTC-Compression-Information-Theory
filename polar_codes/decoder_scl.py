"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import sc_decode, _bit_reversed, _active_llr_level, _active_bit_level, upper_llr, lower_llr


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_step(reg, bit, poly, crc_length):
    reg ^= int(bit) << (crc_length - 1)
    for _ in range(8):
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """CRC-8 / CRC-16，附加在校验比特之后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for b in info_bits:
        reg = _crc_step(reg, b, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for b in bits:
        reg = _crc_step(reg, b, poly, crc_length)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（路径复制 + PM 裁剪）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self._frozen_set = set(np.where(self.frozen_bits)[0])

    @staticmethod
    def _pm_add(pm, llr, u):
        hard = 0 if llr >= 0 else 1
        return pm + (0.0 if u == hard else abs(llr))

    def _init_path(self, llr_ch):
        return {
            "pm": 0.0,
            "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "B": np.full((self.N, self.n + 1), np.nan),
            "u_hat": np.zeros(self.N, dtype=int),
        }

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        B = path["B"]
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [self._init_path(llr_ch)]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_phi = path["L"][l, self.n]

                if l in self._frozen_set:
                    pm = self._pm_add(path["pm"], llr_phi, 0)
                    child = copy.deepcopy(path)
                    child["pm"] = pm
                    child["u_hat"][l] = 0
                    child["B"][l, self.n] = 0
                    self._update_bits(child, l)
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        pm = self._pm_add(path["pm"], llr_phi, u)
                        child = copy.deepcopy(path)
                        child["pm"] = pm
                        child["u_hat"][l] = u
                        child["B"][l, self.n] = u
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            for p in paths:
                bits = p["u_hat"][self.info_indices]
                if crc_check(bits, self.crc_length):
                    return p["u_hat"].copy(), p["pm"]
        best = paths[0]
        return best["u_hat"].copy(), best["pm"]
