"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _SCDCore,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位附加（标准多项式）"""
    info_bits = np.asarray(info_bits, dtype=np.int64)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        fb = ((reg >> (crc_length - 1)) & 1) ^ int(bit)
        reg = ((reg << 1) & mask) ^ (poly if fb else 0)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int64)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) & 1) ^ int(bit)
        reg = ((reg << 1) & mask) ^ (poly if fb else 0)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（路径复制 + Permuted SC 结构）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        frozen_bits = np.asarray(frozen_bits)
        self.frozen_set = set(
            np.where(frozen_bits.astype(bool) if frozen_bits.dtype != bool else frozen_bits)[0]
        )
        self.info_indices = sorted(set(range(N)) - self.frozen_set)

    def _new_path(self, llr_ch):
        path = {
            "pm": 0.0,
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=int),
            "u_hat": np.zeros(self.N, dtype=int),
        }
        path["L"][:, 0] = llr_ch
        return path

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        B = path["B"]
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]

        paths = [self._new_path(llr_ch)]
        order = [_bit_reversed(i, self.n) for i in range(self.N)]

        for l in order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path["L"][l, self.n]
                if l in self.frozen_set:
                    penalty = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    child = self._clone_path(path)
                    child["pm"] += penalty
                    child["B"][l, self.n] = 0
                    child["u_hat"][l] = 0
                    self._update_bits(child, l)
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        child = self._clone_path(path)
                        if (bit == 0 and llr_bit < 0) or (bit == 1 and llr_bit >= 0):
                            child["pm"] += abs(llr_bit)
                        child["B"][l, self.n] = bit
                        child["u_hat"][l] = bit
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            for p in paths:
                info_bits = p["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return p["u_hat"].copy(), p["pm"]
        best = paths[0]
        return best["u_hat"].copy(), best["pm"]

    @staticmethod
    def _clone_path(path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "u_hat": path["u_hat"].copy(),
        }
