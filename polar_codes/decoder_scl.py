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
    f_operation,
    g_operation,
    precompute_sc_indices,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if crc_length == 8:
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
            else:
                if reg & (1 << (crc_length - 1)):
                    reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
                else:
                    reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8 (0x07) 或 CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        rem = _crc_remainder(info_bits, _CRC8_POLY, 8)
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        rem = 0
        poly = _CRC16_POLY
        for b in info_bits:
            rem ^= int(b) << 15
            for _ in range(8):
                if rem & 0x8000:
                    rem = ((rem << 1) ^ poly) & 0xFFFF
                else:
                    rem = (rem << 1) & 0xFFFF
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _path_metric_penalty(llr, u):
    """路径度量惩罚：与硬判决不一致时加 |LLR|。"""
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


# ==================== SCL 译码器 ====================


class _Path:
  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n):
      self.L = np.zeros((N, n + 1), dtype=np.float64)
      self.B = np.zeros((N, n + 1), dtype=int)
      self.pm = 0.0
      self.u_hat = np.zeros(N, dtype=int)

  def copy(self):
      p = _Path(len(self.u_hat), self.L.shape[1] - 1)
      p.L = self.L.copy()
      p.B = self.B.copy()
      p.pm = self.pm
      p.u_hat = self.u_hat.copy()
      return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy：分裂时复制路径）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order, _, _, _ = precompute_sc_indices(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        top_bit,
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] + path.B[j - branch_size, s]
                    ) % 2
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        init = _Path(self.N, self.n)
        init.L[:, 0] = llr_ch
        paths = [init]

        for phi in range(self.N):
            l = self.decode_order[phi]
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    new_path = path.copy()
                    penalty = _path_metric_penalty(llr, 0)
                    new_path.pm += penalty
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = path.copy()
                        penalty = _path_metric_penalty(llr, u)
                        new_path.pm += penalty
                        new_path.B[l, self.n] = u
                        new_path.u_hat[l] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
