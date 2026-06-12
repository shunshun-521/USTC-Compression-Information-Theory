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
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits.astype(int), crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length)[-crc_length:], bits[-crc_length:])


class _Path:
  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n, llr_ch):
      self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
      self.B = np.full((N, n + 1), np.nan)
      self.L[:, 0] = llr_ch
      self.pm = 0.0
      self.u_hat = np.zeros(N, dtype=int)

  def copy(self):
      p = _Path.__new__(_Path)
      p.L = np.copy(self.L)
      p.B = np.copy(self.B)
      p.pm = self.pm
      p.u_hat = self.u_hat.copy()
      return p


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_indices = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _metric(pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def _update_llrs(self, path, bit_idx):
        L, B = path.L, path.B
        n = self.n
        for stage in range(n - _active_llr_level(bit_idx, n), n):
            block_size = 2 ** (stage + 1)
            branch_size = block_size // 2
            for j in range(bit_idx, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, stage + 1] = f_operation(L[j, stage], L[j + branch_size, stage])
                else:
                    L[j, stage + 1] = g_operation(
                        L[j - branch_size, stage],
                        L[j, stage],
                        B[j - branch_size, stage + 1],
                    )
        return L[bit_idx, n]

    def _update_bits(self, path, bit_idx):
        B = path.B
        n = self.n
        if bit_idx < self.N // 2:
            return
        for stage in range(n, n - _active_bit_level(bit_idx, n), -1):
            block_size = 2 ** stage
            branch_size = block_size // 2
            for j in range(bit_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, stage - 1] = int(B[j, stage]) ^ int(B[j - branch_size, stage])
                    B[j, stage - 1] = B[j, stage]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for bit_idx in [_bit_reversed(i, self.n) for i in range(self.N)]:
            candidates = []

            for path in paths:
                llr = self._update_llrs(path, bit_idx)

                if bit_idx in self.frozen_indices:
                    child = path.copy()
                    child.pm = self._metric(path.pm, llr, 0)
                    child.u_hat[bit_idx] = 0
                    child.B[bit_idx, self.n] = 0
                    self._update_bits(child, bit_idx)
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm = self._metric(path.pm, llr, bit)
                        child.u_hat[bit_idx] = bit
                        child.B[bit_idx, self.n] = bit
                        self._update_bits(child, bit_idx)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat, best.pm
