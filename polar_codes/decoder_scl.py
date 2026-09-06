"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _prepare_channel_llr,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        if info_indices is None:
            self.info_indices = np.where(~self.frozen_bits)[0]
        else:
            self.info_indices = np.asarray(info_indices)

    def _update_llrs(self, paths, l):
        for path in paths:
            for s in range(self.n - _active_llr_level(l, self.n), self.n):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                    else:
                        top_bit = int(path.B[j - branch_size, s + 1])
                        path.L[j, s + 1] = g_operation(
                            path.L[j - branch_size, s],
                            path.L[j, s],
                            top_bit,
                        )

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

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr = _prepare_channel_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr.copy())]

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            self._update_llrs(paths, l)

            candidates = []
            for pidx, path in enumerate(paths):
                llr_val = path.L[l, self.n]
                if l in self.frozen_set:
                    penalty = abs(llr_val) if llr_val < 0 else 0.0
                    candidates.append((path.pm + penalty, pidx, 0))
                else:
                    for bit in (0, 1):
                        expected = 0 if llr_val >= 0 else 1
                        penalty = 0.0 if bit == expected else abs(llr_val)
                        candidates.append((path.pm + penalty, pidx, bit))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, pidx, bit in candidates:
                parent = paths[pidx]
                child = _Path(self.N, self.n, parent.L[:, 0].copy())
                child.pm = pm
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()
                child.u_hat[l] = bit
                child.B[l, self.n] = bit
                self._update_bits(child, l)
                new_paths.append(child)

            paths = new_paths

        if self.crc_length > 0:
            crc_pass = [
                i for i, p in enumerate(paths)
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(crc_pass or range(len(paths)), key=lambda i: paths[i].pm)
        else:
            best = min(range(len(paths)), key=lambda i: paths[i].pm)

        return paths[best].u_hat.copy(), paths[best].pm
