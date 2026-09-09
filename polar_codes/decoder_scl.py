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
    precompute_sc_indices,
)


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """
    SCL 译码器（Vangala 置换 SC + Lazy Copy）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        _, _, _, self.phase_order = precompute_sc_indices(N)

    def _pm_penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit == hard else abs(llr_val)

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 1 << (s + 1)
            branch = block // 2
            for j in range(l, self.N, block):
                if j % block < branch:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch, s], int(B[j - branch, s + 1])
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        L, B = path["L"], path["B"]
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 1 << s
            branch = block // 2
            for j in range(l, -1, -block):
                if j % block >= branch:
                    B[j - branch, s - 1] = int(B[j, s]) ^ int(B[j - branch, s])
                    B[j, s - 1] = B[j, s]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        L[:, 0] = llr_ch
        return {
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=np.int8),
            "L": L,
            "B": B,
        }

    def _fork_path(self, parent):
        return {
            "pm": parent["pm"],
            "u_hat": parent["u_hat"].copy(),
            "L": parent["L"].copy(),
            "B": parent["B"].copy(),
        }

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for l in self.phase_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr0 = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    child = self._fork_path(path)
                    child["pm"] += self._pm_penalty(llr0, 0)
                    child["u_hat"][l] = 0
                    child["B"][l, self.n] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = self._fork_path(path)
                        child["pm"] += self._pm_penalty(llr0, bit)
                        child["u_hat"][l] = bit
                        child["B"][l, self.n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u_hat"], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
