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
    _permute_llr,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


# ==================== SCL 译码器 ====================


class _Path:
  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n, llr_ch):
      self.L = np.zeros((N, n + 1), dtype=np.float64)
      self.B = np.zeros((N, n + 1), dtype=int)
      self.L[:, 0] = llr_ch
      self.pm = 0.0
      self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _permute_llr(llr_ch, self.N)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phase in range(self.N):
            l = _bit_reversed(phase, self.n)
            is_frozen = l in self.frozen_set
            candidates = []

            for pidx, path in enumerate(paths):
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if is_frozen:
                    pen = self._path_metric_penalty(llr, 0)
                    new_path = self._clone_path(path)
                    new_path.pm += pen
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append((new_path.pm, pidx, new_path))
                else:
                    for bit in (0, 1):
                        new_path = self._clone_path(path)
                        new_path.pm += self._path_metric_penalty(llr, bit)
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append((new_path.pm, pidx, new_path))

            candidates.sort(key=lambda x: x[0])
            paths = [c[2] for c in candidates[: self.list_size]]

        best = self._select_best_path(paths)
        return best.u_hat.copy(), best.pm

    def _clone_path(self, path):
        new = _Path(self.N, self.n, path.L[:, 0])
        new.L[:] = path.L
        new.B[:] = path.B
        new.pm = path.pm
        new.u_hat[:] = path.u_hat
        return new

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

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            valid = []
            for p in paths:
                payload = p.u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                return min(valid, key=lambda p: p.pm)
        return min(paths, key=lambda p: p.pm)
