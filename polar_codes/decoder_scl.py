"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), poly, crc_length
    )
    crc_bits = np.array(
        [(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（置换 SC + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _new_path(self, llr):
        return {
            "pm": 0.0,
            "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "B": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "u_hat": np.zeros(self.N, dtype=int),
        }

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path["L"][j, s + 1] = f_operation(
                        path["L"][j, s], path["L"][j + branch_size, s]
                    )
                else:
                    top_bit = path["B"][j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    path["L"][j, s + 1] = g_operation(
                        path["L"][j - branch_size, s],
                        path["L"][j, s],
                        int(top_bit),
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path["B"][j - branch_size, s - 1] = (
                        int(path["B"][j, s]) + int(path["B"][j - branch_size, s])
                    ) % 2
                    path["B"][j, s - 1] = path["B"][j, s]

    def _pm_update(self, pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]

        paths = [self._new_path(llr)]
        paths[0]["L"][:, 0] = llr

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path["L"][l, self.n]

                if l in self.frozen_set:
                    child = copy.deepcopy(path)
                    child["pm"] = self._pm_update(path["pm"], llr_val, 0)
                    child["u_hat"][l] = 0
                    child["B"][l, self.n] = 0
                    self._update_bits(child, l)
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        child = copy.deepcopy(path)
                        child["pm"] = self._pm_update(path["pm"], llr_val, bit)
                        child["u_hat"][l] = bit
                        child["B"][l, self.n] = bit
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            info_positions = sorted(set(range(self.N)) - self.frozen_set)
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][info_positions], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p["pm"])

        return best["u_hat"], best["pm"]
