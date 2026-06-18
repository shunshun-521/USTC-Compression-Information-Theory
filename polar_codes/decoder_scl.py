"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_bits(info_bits, poly, crc_length):
    reg = 0
    top = 1 << crc_length
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & (top - 1)
        else:
            reg = (reg << 1) & (top - 1)
    return np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    crc = _crc_bits(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    payload = bits[:-crc_length]
    expected = _crc_bits(payload, poly, crc_length)
    return np.array_equal(bits[-crc_length:], expected)


def _path_metric_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if hard == bit else abs(llr)


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n, llr_ch, parent=None):
        if parent is None:
            self.L = np.zeros((N, n + 1), dtype=np.float64)
            self.L[:, 0] = llr_ch
            self.B = np.zeros((N, n + 1), dtype=np.int8)
            self.pm = 0.0
        else:
            self.L = parent.L.copy()
            self.B = parent.B.copy()
            self.pm = parent.pm


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _extract_u(self, path):
        u = np.zeros(self.N, dtype=int)
        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            u[phi] = path.B[l, self.n]
        return u

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_leaf = path.L[l, self.n]

                if self.frozen_bits[phi]:
                    candidates.append((path.pm + _path_metric_penalty(llr_leaf, 0), path, 0))
                else:
                    for bit in (0, 1):
                        pm = path.pm + _path_metric_penalty(llr_leaf, bit)
                        candidates.append((pm, path, bit))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, parent, bit in candidates[: self.list_size]:
                child = _Path(self.N, self.n, llr_ch, parent=parent)
                child.pm = pm
                child.B[l, self.n] = bit
                self._update_bits(child, l)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u = self._extract_u(p)
                payload = u[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append((p.pm, u))
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        best = min(paths, key=lambda p: p.pm)
        return self._extract_u(best), best.pm
