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
    _get_decode_order,
    f_operation,
    g_operation,
)


_CRC_POLYS = {
    8: np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8),
    16: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int8),
}


def _crc_mod(bits, poly):
    r = len(poly) - 1
    reg = np.zeros(r, dtype=np.int8)
    for bit in bits:
        feedback = (bit ^ reg[0]) & 1
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg ^= poly[1:]
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC_POLYS[crc_length]
    remainder = _crc_mod(info_bits, poly)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC_POLYS[crc_length]
    remainder = _crc_mod(bits, poly)
    return np.all(remainder == 0)


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class _Path:
    __slots__ = ("pm", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        p = _Path.__new__(_Path)
        p.pm = self.pm
        p.B = self.B.copy()
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Permuted SCD + Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = _get_decode_order(N)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, L, B, l):
        start_s = self.n - _active_llr_level(l, self.n)
        for s in range(start_s, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        end_s = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, end_s, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        L[:, 0] = llr_ch

        paths = [_Path(self.N, self.n)]

        for l in self.decode_order:
            for path in paths:
                self._update_llrs(L, path.B, l)

            llr0 = L[l, self.n]
            candidates = []

            if l in self.frozen_set:
                for path in paths:
                    path.pm = _pm_update(path.pm, llr0, 0)
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path.B, l)
                    candidates.append(path)
            else:
                for path in paths:
                    for u in (0, 1):
                        p = path.copy()
                        p.pm = _pm_update(p.pm, llr0, u)
                        p.B[l, self.n] = u
                        p.u_hat[l] = u
                        self._update_bits(p.B, l)
                        candidates.append(p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = min(paths, key=lambda p: p.pm)

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
