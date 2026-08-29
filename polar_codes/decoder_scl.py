"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
  _active_bit_level,
  _active_llr_level,
  _lower_llr,
  _upper_llr,
  bit_reversed,
)
from encoder import bit_reversal_permutation


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
  reg = 0
  for bit in bits:
    reg = ((reg << 1) | int(bit)) & ((1 << (crc_length + 1)) - 1)
    if reg & (1 << crc_length):
      reg ^= poly
  for _ in range(crc_length):
    reg <<= 1
    if reg & (1 << crc_length):
      reg ^= poly
  return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后。"""
  info_bits = np.asarray(info_bits, dtype=int)
  poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
  remainder = _crc_remainder(info_bits, poly, crc_length)
  crc_bits = np.array(
    [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
    dtype=int,
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits 是否通过 CRC。"""
  bits = np.asarray(bits, dtype=int)
  poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
  return _crc_remainder(bits, poly, crc_length) == 0


class Path:
  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n):
    self.L = np.zeros((N, n + 1), dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=np.int8)
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
  """SCL 译码器（路径复制实现）。"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.info_positions = np.where(~self.frozen_bits)[0]
    self.br = bit_reversal_permutation(N)

  def _update_llrs(self, L, B, l):
    n = self.n
    N = self.N
    for s in range(n - _active_llr_level(l, n), n):
      block_size = 2 ** (s + 1)
      branch_size = block_size // 2
      for j in range(l, N, block_size):
        if j % block_size < branch_size:
          L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
        else:
          btm = L[j, s]
          top = L[j - branch_size, s]
          top_bit = B[j - branch_size, s + 1]
          L[j, s + 1] = _lower_llr(btm, top, top_bit)

  def _update_bits(self, B, l):
    n = self.n
    N = self.N
    if l < N // 2:
      return
    for s in range(n, n - _active_bit_level(l, n), -1):
      block_size = 2 ** s
      branch_size = block_size // 2
      for j in range(l, -1, -block_size):
        if j % block_size >= branch_size:
          B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
          B[j, s - 1] = B[j, s]

  def _copy_path(self, path):
    new = Path(self.N, self.n)
    new.L = path.L.copy()
    new.B = path.B.copy()
    new.pm = path.pm
    new.u_hat = path.u_hat.copy()
    return new

  @staticmethod
  def _penalty(llr_val, u_bit):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr_val)

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    paths = [Path(self.N, self.n)]
    paths[0].L[:, 0] = llr_ch[self.br]

    for i in range(self.N):
      l = bit_reversed(i, self.n)
      candidates = []

      for path in paths:
        self._update_llrs(path.L, path.B, l)
        llr_val = path.L[l, self.n]

        if self.frozen_bits[l]:
          new_path = self._copy_path(path)
          new_path.pm += self._penalty(llr_val, 0)
          new_path.u_hat[l] = 0
          new_path.B[l, self.n] = 0
          self._update_bits(new_path.B, l)
          candidates.append(new_path)
        else:
          for u_bit in (0, 1):
            new_path = self._copy_path(path)
            new_path.pm += self._penalty(llr_val, u_bit)
            new_path.u_hat[l] = u_bit
            new_path.B[l, self.n] = u_bit
            self._update_bits(new_path.B, l)
            candidates.append(new_path)

      candidates.sort(key=lambda p: p.pm)
      paths = candidates[: self.list_size]

    if self.crc_length > 0:
      valid = [
        p
        for p in paths
        if crc_check(p.u_hat[self.info_positions], self.crc_length)
      ]
      if valid:
        best = min(valid, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm

    best = min(paths, key=lambda p: p.pm)
    return best.u_hat.copy(), best.pm
