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
    sc_decode,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位。"""
    info_bits = np.asarray(info_bits, dtype=int)
    digest = _crc_digest(info_bits, crc_length)
    crc_bits = np.array(
        [(digest >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def _crc_digest(info_bits, crc_length):
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_check(bits, crc_length=8):
    """检验 CRC（与 crc_encode 成对使用）。"""
    bits = np.asarray(bits, dtype=int)
    if bits.size < crc_length:
        return False
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（置换 SC + 路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s], path.L[j - branch_size, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            u = sc_decode(llr_ch, self.frozen_bits.astype(int))
            return u, 0.0

        paths = [_Path(self.N, self.n, llr_ch)]

        for phi_nat in range(self.N):
            l = _bit_reversed(phi_nat, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    bit_opts = [(0, path.pm + (0.0 if llr_val >= 0 else abs(llr_val)))]
                else:
                    pm0 = path.pm + (0.0 if llr_val >= 0 else abs(llr_val))
                    pm1 = path.pm + (0.0 if llr_val < 0 else abs(llr_val))
                    bit_opts = [(0, pm0), (1, pm1)]

                for u_bit, pm in bit_opts:
                    p = _Path(self.N, self.n, llr_ch)
                    p.L = path.L.copy()
                    p.B = path.B.copy()
                    p.u_hat = path.u_hat.copy()
                    p.pm = pm
                    p.B[l, self.n] = 0 if l in self.frozen_set else u_bit
                    p.u_hat[l] = p.B[l, self.n]
                    self._update_bits(p, l)
                    candidates.append(p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_idx], self.crc_length)
            ]
            best = valid[0] if valid else paths[0]
        else:
            best = paths[0]
        return best.u_hat, best.pm
