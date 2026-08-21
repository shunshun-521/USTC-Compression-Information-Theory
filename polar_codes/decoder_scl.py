"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _align_llr_for_decoder,
    active_bit_level,
    active_llr_level,
    bit_reversed_value,
    hard_decision,
    lower_llr,
    upper_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).tolist()
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= bit << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.array(info_bits + crc_bits, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class PathState:
    """单条 SCL 路径状态。"""

    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_aligned):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_aligned

    def clone(self):
        new_path = PathState.__new__(PathState)
        new_path.pm = self.pm
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        return new_path


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0].tolist())
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0.0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = lower_llr(path.L[j, s], path.L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        if l >= self.N // 2:
            for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size >> 1
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                            path.B[j - branch_size, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_aligned = _align_llr_for_decoder(llr_ch)
        paths = [PathState(self.N, self.n, llr_aligned)]
        decode_order = [bit_reversed_value(i, self.n) for i in range(self.N)]

        for l in decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    cand = path.clone()
                    cand.pm += self._path_metric_penalty(llr, 0)
                    cand.B[l, self.n] = 0
                    self._update_bits(cand, l)
                    new_paths.append(cand)
                else:
                    for u_bit in (0, 1):
                        cand = path.clone()
                        cand.pm += self._path_metric_penalty(llr, u_bit)
                        cand.B[l, self.n] = u_bit
                        self._update_bits(cand, l)
                        new_paths.append(cand)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = path.B[:, self.n].astype(int)
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        u_hat = best.B[:, self.n].astype(int)
        return u_hat, best.pm
