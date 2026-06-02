"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    upper_llr,
    lower_llr,
    active_llr_level,
    active_bit_level,
    _channel_to_natural_llr,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


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
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


# ==================== SCL 译码器 ====================


class PathState:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_indices = list(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def _clone(self, path):
        child = PathState(self.N, self.n)
        child.pm = path.pm
        child.L = path.L.copy()
        child.B = path.B.copy()
        return child

    def decode(self, llr_ch):
        llr_nat = _channel_to_natural_llr(np.asarray(llr_ch, dtype=np.float64))
        N, n = self.N, self.n
        frozen_set = set(self.frozen_indices)

        paths = [PathState(N, n)]
        paths[0].L[:, 0] = llr_nat

        for i in range(N):
            l = int(format(i, f"0{n}b")[::-1], 2)

            for path in paths:
                for s in range(n - active_llr_level(l, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                        else:
                            path.L[j, s + 1] = lower_llr(
                                path.L[j, s], path.L[j - branch_size, s], path.B[j - branch_size, s + 1]
                            )

            if l in frozen_set:
                for path in paths:
                    path.pm += self._pm_penalty(path.L[l, n], 0)
                    path.B[l, n] = 0
                    self._update_bits(path, l)
            else:
                new_paths = []
                for path in paths:
                    llr_val = path.L[l, n]
                    for u_bit in (0, 1):
                        child = self._clone(path)
                        child.pm += self._pm_penalty(llr_val, u_bit)
                        child.B[l, n] = u_bit
                        self._update_bits(child, l)
                        new_paths.append(child)
                new_paths.sort(key=lambda p: p.pm)
                paths = new_paths[: self.list_size]

        best = self._select_path(paths)
        return best.B[:, n].astype(int), best.pm

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        n = self.n
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _select_path(self, paths):
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.B[:, self.n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                return min(valid, key=lambda p: p.pm)
        return min(paths, key=lambda p: p.pm)
