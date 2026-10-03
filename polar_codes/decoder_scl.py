"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reverse,
    _active_llr_level,
    _active_bit_level,
    _hard_decision,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    top = 1 << crc_length
    for b in bits:
        reg <<= 1
        if b:
            reg |= 1
        if reg & top:
            reg ^= poly
    return reg & (top - 1)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _PathState:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch


class SCLDecoder:
    """SCL 译码器（叶节点×阶段存储，路径分裂时复制状态）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reverse(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l_idx):
        for s in range(self.n - _active_llr_level(l_idx, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l_idx, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l_idx):
        if l_idx < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l_idx, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u):
        v = _hard_decision(llr)
        return 0.0 if u == v else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.N, self.n, llr_ch)]

        for l_idx in self.decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l_idx)
                llr = path.L[l_idx, self.n]
                if l_idx in self.frozen_set:
                    pen = self._pm_penalty(llr, 0)
                    new = _PathState(self.N, self.n, llr_ch)
                    new.pm = path.pm + pen
                    new.L = path.L.copy()
                    new.B = path.B.copy()
                    new.B[l_idx, self.n] = 0
                    candidates.append(new)
                else:
                    for u in (0, 1):
                        pen = self._pm_penalty(llr, u)
                        new = _PathState(self.N, self.n, llr_ch)
                        new.pm = path.pm + pen
                        new.L = path.L.copy()
                        new.B = path.B.copy()
                        new.B[l_idx, self.n] = u
                        candidates.append(new)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.L]
            for path in paths:
                self._update_bits(path, l_idx)

        if self.crc_length > 0:
            valid = []
            for i, p in enumerate(paths):
                u = p.B[:, self.n].astype(int)
                info_bits = u[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append((p.pm, i))
            if valid:
                best = paths[min(valid, key=lambda x: x[0])[1]]
            else:
                best = paths[0]
        else:
            best = paths[0]

        return best.B[:, self.n].astype(int), best.pm
