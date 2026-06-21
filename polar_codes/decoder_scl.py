"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from channel import channel_llr_to_decode
from decoder_sc import (
  _bit_reversed,
  _update_bits,
  _update_llrs,
)


def _crc_poly(crc_length):
  if crc_length == 8:
    return 0x07
  if crc_length == 16:
    return 0x8005
  raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后"""
  info_bits = np.asarray(info_bits, dtype=np.int8)
  poly = _crc_poly(crc_length)
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
  crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits 尾部 CRC 是否正确"""
  bits = np.asarray(bits, dtype=np.int8)
  expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
  return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
  """SCL 译码器（Lazy Copy 优化）"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.info_indices = np.where(~self.frozen_bits)[0]

  def decode(self, llr_ch):
    llr_ch = channel_llr_to_decode(np.asarray(llr_ch, dtype=np.float64))
    N, n = self.N, self.n

    paths = [{
      "L": np.zeros((N, n + 1), dtype=np.float64),
      "B": np.zeros((N, n + 1), dtype=np.int8),
      "pm": 0.0,
      "u_hat": np.zeros(N, dtype=np.int8),
    }]
    paths[0]["L"][:, 0] = llr_ch

    for phi in range(N):
      l = _bit_reversed(phi, n)
      candidates = []

      for path in paths:
        _update_llrs(path["L"], path["B"], l, n)
        llr = path["L"][l, n]
        if self.frozen_bits[l]:
          ext = 0
          penalty = 0.0 if llr >= 0 else abs(llr)
          candidates.append((path["pm"] + penalty, path, ext))
        else:
          for ext in (0, 1):
            consistent = (ext == 0 and llr >= 0) or (ext == 1 and llr < 0)
            penalty = 0.0 if consistent else abs(llr)
            candidates.append((path["pm"] + penalty, path, ext))

      candidates.sort(key=lambda x: x[0])
      survivors = candidates[: self.list_size]

      new_paths = []
      for pm, parent, ext in survivors:
        child = {
          "L": parent["L"].copy(),
          "B": parent["B"].copy(),
          "pm": pm,
          "u_hat": parent["u_hat"].copy(),
        }
        child["u_hat"][l] = ext
        child["B"][l, n] = ext
        _update_bits(child["B"], l, n)
        new_paths.append(child)
      paths = new_paths

    if self.crc_length > 0:
      valid = [
        p for p in paths
        if crc_check(p["u_hat"][self.info_indices], self.crc_length)
      ]
      if valid:
        paths = valid

    best = min(paths, key=lambda p: p["pm"])
    return best["u_hat"].astype(int), best["pm"]
