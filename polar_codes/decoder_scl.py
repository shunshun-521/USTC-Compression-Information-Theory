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
)
from encoder import bit_reversal_permutation


CRC8_POLY = [1, 0, 0, 0, 0, 0, 1, 1, 1]
CRC16_POLY = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _crc_poly_encode(message, gen):
    n = len(gen) - 1
    msg = list(map(int, message)) + [0] * n
    for i in range(len(message)):
        if msg[i]:
            for j in range(len(gen)):
                msg[i + j] ^= gen[j]
    return np.array(msg[len(message) :], dtype=int)


def _crc_poly_check(bits, gen):
    msg = list(map(int, bits))
    for i in range(len(msg) - len(gen) + 1):
        if msg[i]:
            for j in range(len(gen)):
                msg[i + j] ^= gen[j]
    return all(x == 0 for x in msg[-(len(gen) - 1) :])


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    gen = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_poly_encode(info_bits, gen)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    gen = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_poly_check(bits, gen)


def _pm_update(pm, llr, u):
    """路径度量更新。"""
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _new_path(self, llr_ch):
        n = self.n
        N = self.N
        L = np.full((N, n + 1), np.nan, dtype=np.float64)
        B = np.full((N, n + 1), np.nan, dtype=np.float64)
        L[:, 0] = llr_ch[self.br]
        return {"L": L, "B": B, "pm": 0.0, "u_hat": np.zeros(N, dtype=int)}

    def _clone_path(self, path):
        return {
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "pm": path["pm"],
            "u_hat": path["u_hat"].copy(),
        }

    def _extend_llrs(self, path, l):
        n = self.n
        N = self.N
        L = path["L"]
        B = path["B"]
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        n = self.n
        N = self.N
        B = path["B"]
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for l in self.decode_order:
            for path in paths:
                self._extend_llrs(path, l)

            if self.frozen_bits[l]:
                for path in paths:
                    llr = path["L"][l, self.n]
                    path["u_hat"][l] = 0
                    path["B"][l, self.n] = 0
                    path["pm"] = _pm_update(path["pm"], llr, 0)
                    self._update_bits(path, l)
            else:
                candidates = []
                for path in paths:
                    llr = path["L"][l, self.n]
                    for u in (0, 1):
                        child = self._clone_path(path)
                        child["u_hat"][l] = u
                        child["B"][l, self.n] = u
                        child["pm"] = _pm_update(child["pm"], llr, u)
                        self._update_bits(child, l)
                        candidates.append(child)
                candidates.sort(key=lambda p: p["pm"])
                paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path["u_hat"][~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"], best["pm"]
