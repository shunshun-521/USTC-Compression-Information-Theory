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
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int_)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int_,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int_)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _PathState:
    """单条 SCL 路径状态（Lazy Copy 通过共享数组 + 活跃标志实现）。"""

    __slots__ = ("L", "B", "pm", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int_)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.active = True

    def copy(self):
        new_path = _PathState.__new__(_PathState)
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.active = True
        return new_path


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(np.asarray(frozen_bits).astype(bool))[0])
        self.info_pos = np.setdiff1d(np.arange(N), list(self.frozen_set))

    def _path_llr(self, path, phi):
        l = _bit_reversed_index(phi, self.n)
        _update_llrs(path.L, path.B, l, self.n)
        return path.L[l, self.n], l

    def _path_update_bit(self, path, phi, bit):
        l = _bit_reversed_index(phi, self.n)
        path.B[l, self.n] = bit
        _update_bits(path.B, l, self.n)

    def decode(self, llr_ch):
        brp = bit_reversal_permutation(self.N)
        llr = np.asarray(llr_ch, dtype=np.float64)[brp]

        paths = [_PathState(self.N, self.n, llr)]

        for phi in range(self.N):
            candidates = []

            for path in paths:
                if not path.active:
                    continue
                llr_val, l = self._path_llr(path, phi)

                if l in self.frozen_set:
                    penalty = abs(llr_val) if llr_val < 0 else 0.0
                    new_path = path.copy()
                    self._path_update_bit(new_path, phi, 0)
                    new_path.pm += penalty
                    candidates.append(new_path)
                else:
                    hard = 0 if llr_val >= 0 else 1
                    for bit in (0, 1):
                        new_path = path.copy()
                        penalty = 0.0 if bit == hard else abs(llr_val)
                        new_path.pm += penalty
                        self._path_update_bit(new_path, phi, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = path.B[:, self.n].astype(int)
                if crc_check(u_hat[self.info_pos], self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm
