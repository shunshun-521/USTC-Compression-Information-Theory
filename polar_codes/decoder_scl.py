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
    _lower_llr,
    _prepare_llr,
    _upper_llr,
)


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后"""
  info_bits = np.asarray(info_bits, dtype=int)
  poly = 0x07 if crc_length == 8 else 0x8005

  reg = 0
  for bit in info_bits:
    feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
    reg = (reg << 1) & ((1 << crc_length) - 1)
    if feedback:
      reg ^= poly

  crc_bits = np.array(
    [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
    dtype=int,
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
  if crc_length == 0:
    return True
  bits = np.asarray(bits, dtype=int)
  recomputed = crc_encode(bits[:-crc_length], crc_length)
  return np.array_equal(bits[-crc_length:], recomputed[-crc_length:])


def _path_metric_update(pm, llr, bit):
  hard = 0 if llr >= 0 else 1
  if bit != hard:
    return pm + abs(llr)
  return pm


class SCLDecoder:
  """SCL 译码器"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.frozen_indices = set(np.where(self.frozen_bits)[0])

  def decode(self, llr_ch):
    if self.list_size == 1:
      from decoder_sc import sc_decode
      u_hat = sc_decode(llr_ch, self.frozen_bits)
      return u_hat, 0.0

    llr_ch = _prepare_llr(llr_ch)
    n = self.n

    paths = [{
      "L": np.zeros((self.N, n + 1), dtype=np.float64),
      "B": np.zeros((self.N, n + 1), dtype=int),
      "pm": 0.0,
      "u_hat": np.zeros(self.N, dtype=int),
    }]
    paths[0]["L"][:, 0] = llr_ch

    for i in range(self.N):
      l = _bit_reversed(i, n)
      new_paths = []

      for path in paths:
        self._update_llrs(path, l)
        llr = path["L"][l, n]

        if l in self.frozen_indices:
          child = self._copy_path(path)
          child["pm"] = _path_metric_update(path["pm"], llr, 0)
          child["u_hat"][l] = 0
          child["B"][l, n] = 0
          self._update_bits(child, l)
          new_paths.append(child)
        else:
          for bit in (0, 1):
            child = self._copy_path(path)
            child["pm"] = _path_metric_update(path["pm"], llr, bit)
            child["u_hat"][l] = bit
            child["B"][l, n] = bit
            self._update_bits(child, l)
            new_paths.append(child)

      new_paths.sort(key=lambda p: p["pm"])
      paths = new_paths[: self.list_size]

    if self.crc_length > 0:
      info_idx = np.where(~self.frozen_bits)[0]
      valid = [p for p in paths if crc_check(p["u_hat"][info_idx], self.crc_length)]
      if valid:
        paths = valid

    best = min(paths, key=lambda p: p["pm"])
    return best["u_hat"].copy(), best["pm"]

  def _copy_path(self, path):
    return {
      "L": path["L"].copy(),
      "B": path["B"].copy(),
      "pm": path["pm"],
      "u_hat": path["u_hat"].copy(),
    }

  def _update_llrs(self, path, l):
    L, B = path["L"], path["B"]
    n = self.n
    for s in range(n - _active_llr_level(l, n), n):
      block_size = 2 ** (s + 1)
      branch_size = block_size // 2
      for j in range(l, self.N, block_size):
        if j % block_size < branch_size:
          L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
        else:
          L[j, s + 1] = _lower_llr(L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1])

  def _update_bits(self, path, l):
    B = path["B"]
    n = self.n
    if l < self.N / 2:
      return
    for s in range(n, n - _active_bit_level(l, n), -1):
      block_size = 2 ** s
      branch_size = block_size // 2
      for j in range(l, -1, -block_size):
        if j % block_size >= branch_size:
          B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
          B[j, s - 1] = B[j, s]
