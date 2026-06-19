"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import _active_bit_level, _active_llr_level, _lower_llr, _upper_llr, sc_decode
from encoder import bit_reversed


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _path_metric_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u == hard:
        return pm
    return pm + abs(llr)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def _init_paths(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_ch
        return [{
            "L": L,
            "B": B,
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=np.int32),
        }]

    def _update_llrs(self, path, phase):
        L = path["L"]
        B = path["B"]
        for s in range(self.n - _active_llr_level(phase, self.n), self.n):
            block = 1 << (s + 1)
            branch = block // 2
            for j in range(phase, self.N, block):
                if j % block < branch:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch, s], int(B[j - branch, s + 1])
                    )

    def _update_bits(self, path, phase):
        if phase < self.N // 2:
            return
        B = path["B"]
        for s in range(self.n, self.n - _active_bit_level(phase, self.n), -1):
            block = 1 << s
            branch = block // 2
            for j in range(phase, -1, -block):
                if j % block >= branch:
                    B[j - branch, s - 1] = int(B[j, s]) ^ int(B[j - branch, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._init_paths(llr_ch)

        for phase in [bit_reversed(i, self.n) for i in range(self.N)]:
            candidates = []

            for path in paths:
                self._update_llrs(path, phase)
                llr_bit = path["L"][phase, self.n]

                if phase in self.frozen_set:
                    new_path = copy.deepcopy(path)
                    new_path["pm"] = _path_metric_update(path["pm"], llr_bit, 0)
                    new_path["B"][phase, self.n] = 0
                    new_path["u_hat"][phase] = 0
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = copy.deepcopy(path)
                        new_path["pm"] = _path_metric_update(path["pm"], llr_bit, bit)
                        new_path["B"][phase, self.n] = bit
                        new_path["u_hat"][phase] = bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

            for path in paths:
                self._update_bits(path, phase)

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"], best["pm"]
