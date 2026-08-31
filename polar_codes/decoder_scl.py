"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
  _active_bit_level,
  _active_llr_level,
  _update_bits,
  _update_llrs,
  f_operation,
  g_operation,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
  if crc_length == 8:
    return CRC8_POLY
  if crc_length == 16:
    return CRC16_POLY
  raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后。"""
  info_bits = np.asarray(info_bits, dtype=int)
  poly = _crc_poly(crc_length)
  reg = 0
  mask = (1 << crc_length) - 1
  top = 1 << (crc_length - 1)
  for bit in info_bits:
    reg ^= int(bit) << (crc_length - 1)
    if reg & top:
      reg = ((reg << 1) ^ poly) & mask
    else:
      reg = (reg << 1) & mask
  crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
  bits = np.asarray(bits, dtype=int)
  poly = _crc_poly(crc_length)
  reg = 0
  mask = (1 << crc_length) - 1
  top = 1 << (crc_length - 1)
  for bit in bits:
    reg ^= int(bit) << (crc_length - 1)
    if reg & top:
      reg = ((reg << 1) ^ poly) & mask
    else:
      reg = (reg << 1) & mask
  return reg == 0


class _PathState:
  __slots__ = ("pm", "L", "B")

  def __init__(self, N, n, llr_ch):
    self.pm = 0.0
    self.L = np.zeros((N, n + 1), dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=int)
    self.L[:, 0] = llr_ch


class SCLDecoder:
  """SCL 译码器（Permuted SC + Lazy Copy）。"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.info_indices = np.where(~self.frozen_bits)[0]
    self.decode_order = [int(bit_reversal_permutation(N)[i]) for i in range(N)]

  @staticmethod
  def _branch_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N, n = self.N, self.n

    paths = [_PathState(N, n, llr_ch)]

    for l in self.decode_order:
      candidates = []
      for path in paths:
        _update_llrs(path.L, path.B, l, n)
        llr_l = path.L[l, n]

        if self.frozen_bits[l]:
          pm_new = path.pm + self._branch_penalty(llr_l, 0)
          candidates.append((pm_new, path, 0))
        else:
          for bit in (0, 1):
            pm_new = path.pm + self._branch_penalty(llr_l, bit)
            candidates.append((pm_new, path, bit))

      candidates.sort(key=lambda x: x[0])
      selected = candidates[: self.list_size]

      new_paths = []
      for pm_new, parent, bit in selected:
        child = _PathState(N, n, llr_ch)
        child.pm = pm_new
        child.L = parent.L.copy()
        child.B = parent.B.copy()
        child.B[l, n] = 0 if self.frozen_bits[l] else bit
        _update_bits(child.B, l, n, N)
        new_paths.append(child)
      paths = new_paths

    paths.sort(key=lambda p: p.pm)
    best_pm = paths[0].pm
    best_u = paths[0].B[:, n].astype(int)

    if self.crc_length > 0:
      crc_pass = []
      for path in paths:
        u = path.B[:, n].astype(int)
        info_bits = u[self.info_indices]
        if crc_check(info_bits, self.crc_length):
          crc_pass.append((path.pm, u))
      if crc_pass:
        crc_pass.sort(key=lambda x: x[0])
        return crc_pass[0][1], crc_pass[0][0]

    return best_u, best_pm
