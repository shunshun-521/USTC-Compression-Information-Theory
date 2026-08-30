"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Permuted SCD
"""
import math

import numpy as np

from decoder_sc import (
  _active_bit_level,
  _active_llr_level,
  _bit_reversed_index,
  _update_bits,
  _update_llrs,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
  if crc_length == 8:
    return 0x07
  if crc_length == 16:
    return 0x8005
  raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后。"""
  poly = _crc_poly(crc_length)
  info_bits = np.asarray(info_bits, dtype=np.int8)
  reg = 0
  for bit in info_bits:
    msb = (reg >> (crc_length - 1)) & 1
    reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
    if msb ^ int(bit):
      reg ^= poly
  for _ in range(crc_length):
    msb = (reg >> (crc_length - 1)) & 1
    reg = (reg << 1) & ((1 << crc_length) - 1)
    if msb:
      reg ^= poly
  crc_bits = np.array(
    [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
  if crc_length == 0:
    return True
  expected = crc_encode(bits[:-crc_length], crc_length)
  return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class Path:
  """单条 SCL 路径。"""

  __slots__ = ("pm", "u_hat", "L", "B")

  def __init__(self, N, n):
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=np.int8)
    self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
    self.B = np.full((N, n + 1), np.nan, dtype=np.float64)


class SCLDecoder:
  """SCL 译码器（Permuted SCD + Lazy Copy）。"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.frozen_set = set(np.where(self.frozen_bits)[0])
    self.info_indices = np.where(~self.frozen_bits)[0]
    self.decode_order = [
      _bit_reversed_index(i, self.n) for i in range(N)
    ]

  def _pm_penalty(self, llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)

  def _clone_path(self, path):
    new_path = Path(self.N, self.n)
    new_path.pm = path.pm
    new_path.u_hat = path.u_hat.copy()
    new_path.L = path.L.copy()
    new_path.B = path.B.copy()
    return new_path

  def decode(self, llr_ch):
    """主译码函数。"""
    rev = bit_reversal_permutation(self.N)
    llr = np.asarray(llr_ch, dtype=np.float64)[rev]

    paths = [Path(self.N, self.n)]
    paths[0].L[:, 0] = llr

    for l in self.decode_order:
      candidates = []
      for path in paths:
        _update_llrs(path.L, path.B, l, self.n)
        cur_llr = path.L[l, self.n]

        if l in self.frozen_set:
          new_path = self._clone_path(path)
          new_path.pm += self._pm_penalty(cur_llr, 0)
          new_path.u_hat[l] = 0
          new_path.B[l, self.n] = 0
          _update_bits(new_path.B, l, self.n)
          candidates.append(new_path)
        else:
          for bit in (0, 1):
            new_path = self._clone_path(path)
            new_path.pm += self._pm_penalty(cur_llr, bit)
            new_path.u_hat[l] = bit
            new_path.B[l, self.n] = bit
            _update_bits(new_path.B, l, self.n)
            candidates.append(new_path)

      candidates.sort(key=lambda p: p.pm)
      paths = candidates[: self.list_size]

    crc_pass = []
    for path in paths:
      if self.crc_length > 0:
        info_bits = path.u_hat[self.info_indices]
        if crc_check(info_bits, self.crc_length):
          crc_pass.append(path)
      else:
        crc_pass.append(path)

    best = min(crc_pass or paths, key=lambda p: p.pm)
    return best.u_hat.astype(int), best.pm
