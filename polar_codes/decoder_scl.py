"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
  _active_bit_level,
  _active_llr_level,
  f_operation,
  g_operation,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
  reg = 0
  mask = (1 << crc_length) - 1
  for bit in bits:
    reg ^= int(bit) << (crc_length - 1)
    for _ in range(crc_length):
      if reg & (1 << (crc_length - 1)):
        reg = ((reg << 1) ^ poly) & mask
      else:
        reg = (reg << 1) & mask
  return reg


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后。"""
  info_bits = np.asarray(info_bits, dtype=int)
  poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
  remainder = _crc_remainder(info_bits, poly, crc_length)
  crc_bits = np.array(
    [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
    dtype=int,
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits 是否通过 CRC。"""
  bits = np.asarray(bits, dtype=int)
  poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
  return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
  """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）。"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]
    if info_indices is None:
      self.info_indices = np.flatnonzero(~self.frozen_bits)
    else:
      self.info_indices = np.asarray(info_indices, dtype=int)

  def _update_llrs(self, L, B, l):
    start = self.n - _active_llr_level(l, self.n)
    for s in range(start, self.n):
      block_size = 1 << (s + 1)
      branch_size = block_size // 2
      for j in range(l, self.N, block_size):
        if j % block_size < branch_size:
          L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
        else:
          L[j, s + 1] = g_operation(
            L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
          )

  def _update_bits(self, B, l):
    if l < self.N // 2:
      return
    bit_start = self.n - _active_bit_level(l, self.n)
    for s in range(self.n, bit_start, -1):
      block_size = 1 << s
      branch_size = block_size // 2
      for j in range(l, -1, -block_size):
        if j % block_size >= branch_size:
          B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
          B[j, s - 1] = B[j, s]

  @staticmethod
  def _branch_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)

  def decode(self, llr_ch):
    """
    主译码函数。llr_ch 应为比特倒序后的信道 LLR。
  """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    paths = []
    L0 = np.zeros((self.N, self.n + 1), dtype=np.float64)
    B0 = np.zeros((self.N, self.n + 1), dtype=np.int8)
    L0[:, 0] = llr_ch
    paths.append({"pm": 0.0, "L": L0, "B": B0})

    for l in self.decode_order:
      new_paths = []
      for path in paths:
        self._update_llrs(path["L"], path["B"], l)
        llr = path["L"][l, self.n]

        if self.frozen_bits[l]:
          penalty = 0.0 if llr >= 0 else abs(llr)
          path["pm"] += penalty
          path["B"][l, self.n] = 0
          self._update_bits(path["B"], l)
          new_paths.append(path)
        else:
          for bit in (0, 1):
            child = {
              "pm": path["pm"] + self._branch_penalty(llr, bit),
              "L": path["L"].copy(),
              "B": path["B"].copy(),
            }
            child["B"][l, self.n] = bit
            self._update_bits(child["B"], l)
            new_paths.append(child)

      new_paths.sort(key=lambda p: p["pm"])
      paths = new_paths[: self.list_size]

    if self.crc_length > 0:
      valid = [
        p
        for p in paths
        if crc_check(p["B"][:, self.n][self.info_indices], self.crc_length)
      ]
      best = min(valid if valid else paths, key=lambda p: p["pm"])
    else:
      best = min(paths, key=lambda p: p["pm"])

    return best["B"][:, self.n].copy(), best["pm"]
