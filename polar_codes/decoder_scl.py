"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _bit_reversed, _update_bits, _update_llrs


CRC_POLYS = {
  8: 0x07,
  16: 0x8005,
}


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
  """计算 CRC 校验位并附加到信息比特后。"""
  info_bits = np.asarray(info_bits, dtype=int)
  poly = CRC_POLYS[crc_length]
  remainder = _crc_remainder(info_bits, poly, crc_length)
  crc_bits = np.array(
    [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
    dtype=int,
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
  bits = np.asarray(bits, dtype=int)
  expected = crc_encode(bits[:-crc_length], crc_length)
  return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _Path:
  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n, llr_ch):
    self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=int)
    self.L[:, 0] = llr_ch.copy()
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
  """SCL 译码器。"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.frozen_set = set(np.where(self.frozen_bits)[0])
    self.info_indices = np.where(~self.frozen_bits)[0]

  def _path_metric_penalty(self, llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = self.N
    n = self.n

    active = [_Path(N, n, llr_ch)]

    for phi in range(N):
      l = _bit_reversed(phi, n)
      new_paths = []

      for path in active:
        _update_llrs(path.L, path.B, l, n, N)
        llr = path.L[l, n]

        if l in self.frozen_set:
          penalty = self._path_metric_penalty(llr, 0)
          child = _Path(N, n, llr_ch)
          child.L[:] = path.L
          child.B[:] = path.B
          child.pm = path.pm + penalty
          child.u_hat[:] = path.u_hat
          child.B[l, n] = 0
          child.u_hat[l] = 0
          _update_bits(child.B, l, n, N)
          new_paths.append(child)
        else:
          for bit in (0, 1):
            penalty = self._path_metric_penalty(llr, bit)
            child = _Path(N, n, llr_ch)
            child.L[:] = path.L
            child.B[:] = path.B
            child.pm = path.pm + penalty
            child.u_hat[:] = path.u_hat
            child.B[l, n] = bit
            child.u_hat[l] = bit
            _update_bits(child.B, l, n, N)
            new_paths.append(child)

      new_paths.sort(key=lambda p: p.pm)
      active = new_paths[: self.list_size]

    if self.crc_length > 0:
      valid = []
      for path in active:
        info_bits = path.u_hat[self.info_indices]
        if crc_check(info_bits, self.crc_length):
          valid.append(path)
      if valid:
        active = valid

    best = min(active, key=lambda p: p.pm)
    return best.u_hat.copy(), best.pm
