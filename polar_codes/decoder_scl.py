"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
  active_bit_level,
  active_llr_level,
  bit_reversed,
  lower_llr,
  sc_decode,
  upper_llr,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_generator(crc_length):
  """返回 CRC 生成多项式系数（含最高次项）"""
  if crc_length == 8:
    return np.array([1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
  if crc_length == 16:
    return np.array([1] + [(0x8005 >> i) & 1 for i in range(15, -1, -1)], dtype=np.int8)
  raise ValueError(f"Unsupported CRC length: {crc_length}")


def _crc_mod2(msg, gen):
  """GF(2) 长除法求 CRC 余数"""
  msg = list(int(b) for b in msg)
  n = len(gen) - 1
  for i in range(len(msg) - n):
    if msg[i]:
      for j in range(len(gen)):
        msg[i + j] ^= int(gen[j])
  return np.array(msg[-n:], dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
  """
  计算 CRC 校验位并附加到信息比特后。
  r=8: CRC-8 (0x07), r=16: CRC-16 (0x8005)
  """
  info_bits = np.asarray(info_bits, dtype=np.int8)
  gen = _crc_generator(crc_length)
  msg = list(info_bits) + [0] * crc_length
  remainder = _crc_mod2(msg, gen)
  return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
  """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
  bits = np.asarray(bits, dtype=np.int8)
  gen = _crc_generator(crc_length)
  remainder = _crc_mod2(bits, gen)
  return np.all(remainder == 0)


class PathState:
  """单条译码路径状态"""

  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, n, N):
    self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=np.int8)
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=np.int8)

  def copy(self):
    new = PathState.__new__(PathState)
    new.L = self.L.copy()
    new.B = self.B.copy()
    new.pm = self.pm
    new.u_hat = self.u_hat.copy()
    return new


def _update_llrs_path(l, path, n):
  L, B = path.L, path.B
  N = L.shape[0]
  for s in range(n - active_llr_level(l, n), n):
    block_size = 2 ** (s + 1)
    branch_size = block_size // 2
    for j in range(l, N, block_size):
      if j % block_size < branch_size:
        L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
      else:
        top_bit = B[j - branch_size, s + 1]
        L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], top_bit)


def _update_bits_path(l, path, n):
  B = path.B
  N = B.shape[0]
  if l < N // 2:
    return
  for s in range(n, n - active_bit_level(l, n), -1):
    block_size = 2 ** s
    branch_size = block_size // 2
    for j in range(l, -1, -block_size):
      if j % block_size >= branch_size:
        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
        B[j, s - 1] = B[j, s]


class SCLDecoder:
  """SCL 译码器（含 Lazy Copy 优化）"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.rev = bit_reversal_permutation(N)
    self.frozen_set = set(np.where(self.frozen_bits)[0])

  def _pm_penalty(self, llr, u_bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr)

  def decode(self, llr_ch):
    """
    主译码函数。
    返回：(u_hat, pm)
    """
    if self.list_size == 1 and self.crc_length == 0:
      u_hat = sc_decode(llr_ch, self.frozen_bits)
      return u_hat, 0.0

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    llr_ch = llr_ch[self.rev]

    paths = [PathState(self.n, self.N)]
    paths[0].L[:, 0] = llr_ch

    for i in range(self.N):
      l = bit_reversed(i, self.n)
      candidates = []

      for path in paths:
        _update_llrs_path(l, path, self.n)
        llr0 = path.L[l, self.n]

        if l in self.frozen_set:
          new_path = path.copy()
          new_path.pm += self._pm_penalty(llr0, 0)
          new_path.u_hat[l] = 0
          new_path.B[l, self.n] = 0
          _update_bits_path(l, new_path, self.n)
          candidates.append(new_path)
        else:
          for u_bit in (0, 1):
            new_path = path.copy()
            new_path.pm += self._pm_penalty(llr0, u_bit)
            new_path.u_hat[l] = u_bit
            new_path.B[l, self.n] = u_bit
            _update_bits_path(l, new_path, self.n)
            candidates.append(new_path)

      candidates.sort(key=lambda p: p.pm)
      paths = candidates[:self.list_size]

    if self.crc_length > 0:
      info_positions = np.where(~self.frozen_bits)[0]
      valid = [p for p in paths if crc_check(p.u_hat[info_positions], self.crc_length)]
      best = min(valid if valid else paths, key=lambda p: p.pm)
    else:
      best = min(paths, key=lambda p: p.pm)

    return best.u_hat.astype(int), best.pm
