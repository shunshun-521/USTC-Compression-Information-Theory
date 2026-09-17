"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 PSCD 顺序
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    f_operation,
    g_operation,
)

POLY8 = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
POLY16 = np.array(
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8
)


def _crc_remainder(bits, poly):
    reg = np.zeros(len(poly) - 1, dtype=np.int8)
    for bit in bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg ^= poly[1:]
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = POLY8 if crc_length == 8 else POLY16
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    remainder = _crc_remainder(info_bits, poly)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = POLY8 if crc_length == 8 else POLY16
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


class Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch, pm=0.0):
        self.pm = pm
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [
            _bit_reversed_index(i, self.n) for i in range(N)
        ]

    def _copy_path(self, path):
        new_path = Path(self.N, self.n, path.L[:, 0], path.pm)
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _update_llrs(self, path, l):
        L = path.L
        B = path.B
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l, bit):
        B = path.B
        path.u_hat[l] = bit
        B[l, self.n] = bit
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch, 0.0)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    new_path.pm += self._path_metric_penalty(llr, 0)
                    self._update_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._path_metric_penalty(llr, bit)
                        self._update_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_mask = self.frozen_bits == 0
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[info_mask], self.crc_length)
            ]
            best = min(valid or paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
