"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），PSC 结构
"""
import math

import numpy as np

from decoder_sc import (
    _PSCState,
    _active_bit_level,
    _active_llr_level,
    _lower_llr,
    _upper_llr,
    precompute_sc_indices,
)
from encoder import bit_reversed_index


CRC_POLYS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = CRC_POLYS[crc_length]
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = 0
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    for bit in msg:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class _SCLPath:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    @staticmethod
    def _branch_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
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
        paths = [_SCLPath(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            candidates = []
            for pidx, path in enumerate(paths):
                self._update_llrs(path, l)
                llr_leaf = path.L[l, self.n]
                if self.frozen_bits[l]:
                    bit = 0
                    pm = path.pm + self._branch_penalty(llr_leaf, bit)
                    candidates.append((pm, pidx, bit))
                else:
                    for bit in (0, 1):
                        pm = path.pm + self._branch_penalty(llr_leaf, bit)
                        candidates.append((pm, pidx, bit))

            candidates.sort(key=lambda x: x[0])
            survivors = candidates[: self.list_size]

            new_paths = []
            for pm, parent_idx, bit in survivors:
                child = _SCLPath(self.N, self.n, llr_ch)
                child.L = paths[parent_idx].L.copy()
                child.B = paths[parent_idx].B.copy()
                child.pm = pm
                child.u_hat = paths[parent_idx].u_hat.copy()
                child.u_hat[l] = bit
                child.B[l, self.n] = bit
                self._update_bits(child, l)
                new_paths.append(child)
            paths = new_paths

        crc_pass = []
        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append((path.pm, path))
            else:
                crc_pass.append((path.pm, path))

        if crc_pass:
            _, best = min(crc_pass, key=lambda x: x[0])
        else:
            _, best = min(((p.pm, p) for p in paths), key=lambda x: x[0])

        return best.u_hat.copy(), best.pm
