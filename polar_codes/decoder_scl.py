"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    f_operation,
    g_operation,
)
from encoder import bit_reversed_index


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
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
    recomputed = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], recomputed[-crc_length:])


class _Path:
  __slots__ = ("L", "B", "pm", "u_hat", "active")

  def __init__(self, N, n, llr_ch):
    self.L = np.zeros((N, n + 1), dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=int)
    self.L[:, 0] = llr_ch
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=int)
    self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = {i for i in range(N) if self.frozen_bits[i]}
        self.list_size = list_size
        self.crc_length = crc_length

    def _clone_path(self, path):
        new_path = _Path(self.N, self.n, path.L[:, 0])
        new_path.L[:] = path.L
        new_path.B[:] = path.B
        new_path.pm = path.pm
        new_path.u_hat[:] = path.u_hat
        return new_path

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top = path.L[j - branch_size, s]
                    btm = path.L[j, s]
                    path.L[j, s + 1] = g_operation(top, btm, path.B[j - branch_size, s + 1])

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = bit_reversed_index(phi, self.n)
            candidates = []

            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    pen = self._path_metric_penalty(llr, 0)
                    new_path = self._clone_path(path)
                    new_path.pm += pen
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        pen = self._path_metric_penalty(llr, bit)
                        new_path = self._clone_path(path)
                        new_path.pm += pen
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        crc_paths = []
        if self.crc_length > 0:
            info_idx = np.where(self.frozen_bits == 0)[0]
            for path in paths:
                payload = path.u_hat[info_idx]
                if crc_check(payload, self.crc_length):
                    crc_paths.append(path)

        if crc_paths:
            best = min(crc_paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
