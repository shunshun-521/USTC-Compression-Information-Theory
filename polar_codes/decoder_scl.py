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
    _prepare_channel_llrs,
    f_operation,
    g_operation,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(info_bits, crc_length):
    if crc_length == 8:
        poly = _CRC8_POLY
        width = 8
    elif crc_length == 16:
        poly = _CRC16_POLY
        width = 16
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in np.asarray(info_bits, dtype=int):
        msb = (reg >> (width - 1)) & 1
        reg = (reg << 1) & ((1 << width) - 1)
        if bit ^ msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = _crc_remainder(bits[:-crc_length], crc_length)
    received = 0
    for i, bit in enumerate(bits[-crc_length:]):
        received |= int(bit) << (crc_length - 1 - i)
    return expected == received


def _path_metric_update(pm, llr, bit):
    hard = 0 if llr >= 0 else 1
    penalty = 0.0 if bit == hard else abs(llr)
    return pm + penalty


class _Path:
    __slots__ = ("pm", "L", "B", "active_layers")

    def __init__(self, N, n, channel_llrs):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = channel_llrs.copy()
        self.active_layers = set()


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        if 2 ** self.n != N:
            raise ValueError("N must be a power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _clone_path(self, path):
        new_path = _Path(self.N, self.n, path.L[:, 0])
        new_path.pm = path.pm
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.active_layers = set(path.active_layers)
        return new_path

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            half = block_size // 2
            for j in range(l, self.N, block_size):
                if np.isnan(path.L[j, s + 1]):
                    if j % block_size < half:
                        path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + half, s])
                    else:
                        path.L[j, s + 1] = g_operation(
                            path.L[j - half, s],
                            path.L[j, s],
                            path.B[j - half, s + 1],
                        )

    def _update_bits(self, path, l, bit):
        path.B[l, self.n] = bit
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            half = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= half:
                    path.B[j - half, s - 1] = path.B[j, s] ^ path.B[j - half, s]
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        channel_llrs = _prepare_channel_llrs(llr_ch)
        paths = [_Path(self.N, self.n, channel_llrs)]

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if np.isnan(llr):
                    llr = 0.0

                if l in self.frozen_set:
                    bit = 0
                    new_path = self._clone_path(path)
                    new_path.pm = _path_metric_update(path.pm, llr, bit)
                    self._update_bits(new_path, l, bit)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone_path(path)
                        new_path.pm = _path_metric_update(path.pm, llr, bit)
                        self._update_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc_pm = None
        best_pm = float("inf")
        best_path = paths[0]

        for path in paths:
            u_hat = path.B[:, self.n].astype(int)
            if self.crc_length > 0:
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    if best_crc_pm is None or path.pm < best_crc_pm:
                        best_crc_pm = path.pm
                        best_path = path
            elif path.pm < best_pm:
                best_pm = path.pm
                best_path = path

        if self.crc_length > 0 and best_crc_pm is not None:
            return best_path.B[:, self.n].astype(int), best_crc_pm

        return best_path.B[:, self.n].astype(int), best_path.pm
