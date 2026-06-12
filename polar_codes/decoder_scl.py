"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
  _active_bit_level,
  _active_llr_level,
  _update_bits,
  _update_llr,
  f_operation,
  g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
  """计算 CRC 余数（MSB first）。"""
  reg = 0
  for bit in bits:
    reg ^= int(bit) << (crc_length - 1)
    for _ in range(crc_length):
      if reg & (1 << (crc_length - 1)):
        reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
      else:
        reg = (reg << 1) & ((1 << crc_length) - 1)
  return reg


def crc_encode(info_bits, crc_length=8):
  """
  计算 CRC 校验位并附加到信息比特后。
  """
  poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
  rem = _crc_remainder(info_bits, poly, crc_length)
  crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
  poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
  rem = _crc_remainder(bits[:-crc_length], poly, crc_length)
  expected = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
  return np.array_equal(bits[-crc_length:], expected)


class _Path:
  """单条 SCL 路径（Lazy Copy）。"""

  __slots__ = ("L", "B", "pm", "parent", "copy_id")

  def __init__(self, N, n, llr_ch, parent=None, copy_id=0):
    self.parent = parent
    self.copy_id = copy_id
    if parent is None:
      self.L = np.zeros((N, n + 1), dtype=np.float64)
      self.B = np.zeros((N, n + 1), dtype=np.int32)
      self.L[:, 0] = llr_ch
      self.pm = 0.0
    else:
      self.L = parent.L
      self.B = parent.B
      self.pm = parent.pm

  def get_L(self):
    p = self
    while p.parent is not None and p.copy_id == 0:
      p = p.parent
    return p.L if p.parent is None else p.parent.L

  def get_B(self):
    p = self
    while p.parent is not None and p.copy_id == 0:
      p = p.parent
    return p.B if p.parent is None else p.parent.B


class SCLDecoder:
  """SCL 译码器（含 Lazy Copy 优化）。"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.br = bit_reversal_permutation(N)
    self.info_indices = np.where(~self.frozen_bits)[0]

  def _path_metric_penalty(self, llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)

  def decode(self, llr_ch):
    N, n = self.N, self.n
    br = self.br

    paths = [_Path(N, n, llr_ch)]

    for phi in range(N):
      l = br[phi]
      candidates = []

      for path in paths:
        L = path.L
        B = path.B
        _update_llr(L, B, l, n, N)
        llr = L[l, n]

        if self.frozen_bits[l]:
          pm = path.pm + self._path_metric_penalty(llr, 0)
          candidates.append((pm, path, 0, False))
        else:
          for bit in (0, 1):
            pm = path.pm + self._path_metric_penalty(llr, bit)
            candidates.append((pm, path, bit, True))

      candidates.sort(key=lambda x: x[0])
      selected = candidates[: self.list_size]

      new_paths = []
      for pm, parent, bit, need_copy in selected:
        if need_copy:
          child = _Path(N, n, llr_ch, parent=parent, copy_id=1)
          child.L = parent.L.copy()
          child.B = parent.B.copy()
        else:
          child = _Path(N, n, llr_ch, parent=parent, copy_id=0)
        child.pm = pm
        child.B[l, n] = 0 if self.frozen_bits[l] else bit
        _update_bits(child.B, l, n, N)
        new_paths.append(child)

      paths = new_paths

    best = None
    crc_paths = []
    for path in paths:
      u_hat = path.B[:, n].astype(int)
      if self.crc_length > 0:
        info_bits = u_hat[self.info_indices]
        if crc_check(info_bits, self.crc_length):
          crc_paths.append(path)
      if best is None or path.pm < best.pm:
        best = path

    chosen = min(crc_paths, key=lambda p: p.pm) if crc_paths else best
    u_hat = chosen.B[:, n].astype(int)
    return u_hat, chosen.pm
