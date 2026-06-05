"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from channel import reorder_llr_for_decode
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    expected_crc = _crc_remainder(payload, poly, crc_length)
    received = 0
    for i in range(crc_length):
        received |= int(bits[-(crc_length - i)]) << (crc_length - 1 - i)
    return expected_crc == received


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时共享 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _llr_to_bit(llr_val):
        return 0 if llr_val >= 0 else 1

    def _pm_penalty(self, llr_val, u):
        return 0.0 if u == self._llr_to_bit(llr_val) else abs(llr_val)

    def _compute_llr(self, path, l):
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        n = self.n
        N = self.N
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        N = self.N
        n = self.n
        llr = reorder_llr_for_decode(llr_ch, N)

        root = _Path(N, n)
        root.L[:, 0] = llr
        paths = [root]

        for l in [_bit_reversed(i, n) for i in range(N)]:
            candidates = []

            for path in paths:
                self._compute_llr(path, l)
                cur_llr = path.L[l, n]

                if l in self.frozen_set:
                    child = _Path(N, n)
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.pm = path.pm + self._pm_penalty(cur_llr, 0)
                    child.u_hat = path.u_hat.copy()
                    child.u_hat[l] = 0
                    child.B[l, n] = 0
                    self._update_bits(child, l)
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        child = _Path(N, n)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = path.pm + self._pm_penalty(cur_llr, u)
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[l] = u
                        child.B[l, n] = u
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
