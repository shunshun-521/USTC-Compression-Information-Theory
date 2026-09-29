"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    precompute_sc_indices,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.zeros(crc_length, dtype=np.int8)
    for i in range(crc_length):
        crc_bits[crc_length - 1 - i] = (reg >> i) & 1
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（与 SCD 相同的 LLR/比特树更新）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.phases, _ = precompute_sc_indices(N)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _extend_llr(self, path, l):
        N, n = self.N, self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(
                        path.L[j, s], path.L[j + branch_size, s], min_sum=True
                    )
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        int(path.B[j - branch_size, s + 1]),
                        min_sum=True,
                    )

    def _update_bits(self, path, l):
        N, n = self.N, self.n
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _clone(self, path):
        q = _Path(self.N, self.n, path.L[:, 0])
        q.pm = path.pm
        q.L = path.L.copy()
        q.B = path.B.copy()
        q.u_hat = path.u_hat.copy()
        return q

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.phases:
            candidates = []
            for path in paths:
                self._extend_llr(path, l)
                llr0 = path.L[l, self.n]

                if l in self.frozen_set:
                    path.u_hat[l] = 0
                    if llr0 < 0:
                        path.pm += abs(llr0)
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        child = self._clone(path)
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        hard = 0 if llr0 >= 0 else 1
                        if bit != hard:
                            child.pm += abs(llr0)
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = paths[0]
        best_pm = best.pm
        if self.crc_length > 0:
            for path in paths:
                payload = path.u_hat[self.info_idx]
                if crc_check(payload, self.crc_length) and path.pm < best_pm:
                    best = path
                    best_pm = path.pm

        return best.u_hat.copy(), best_pm
