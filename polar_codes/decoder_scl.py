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
    _to_frozen_indices,
    f_operation,
    g_operation,
    precompute_sc_indices,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_division(info_bits, poly, crc_length):
    """GF(2) CRC 除法，返回余数比特"""
    reg = np.zeros(crc_length, dtype=int)
    poly_bits = [
        (poly >> i) & 1 for i in range(crc_length, -1, -1) if (poly >> i) & 1
    ]
    poly_len = len(poly_bits)
    for bit in info_bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            for i in range(1, poly_len):
                reg[i - 1] ^= poly_bits[i]
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_division(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_division(bits[:-crc_length], poly, crc_length)
    return np.array_equal(remainder, bits[-crc_length:])


class Path:
    """SCL 单条路径"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, n, N):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new_path = Path(self.L.shape[1] - 1, self.L.shape[0])
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(_to_frozen_indices(frozen_bits))
        self.info_positions = np.array(
            sorted(set(range(N)) - self.frozen_set), dtype=int
        )
        self.list_size = list_size
        self.crc_length = crc_length
        precompute_sc_indices(N)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )
        return path.L[l, self.n]

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, u_val):
        hard = 0 if llr >= 0 else 1
        return abs(llr) if u_val != hard else 0.0

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.n, self.N)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                llr = self._update_llrs(path, l)
                if l in self.frozen_set:
                    new_path = path.copy()
                    new_path.pm += self._path_metric_penalty(llr, 0)
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._path_metric_penalty(llr, u_val)
                        new_path.B[l, self.n] = u_val
                        new_path.u_hat[l] = u_val
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_positions], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat, best.pm
