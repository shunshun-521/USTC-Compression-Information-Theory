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


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
  reg = 0
  for bit in bits:
    reg <<= 1
    reg |= int(bit)
    if reg & (1 << crc_length):
      reg ^= poly
  mask = (1 << crc_length) - 1
  return reg & mask


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后。"""
  info_bits = np.asarray(info_bits, dtype=np.int8)
  poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
  remainder = _crc_remainder(info_bits, poly, crc_length)
  crc_bits = np.array(
    [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
    dtype=np.int8,
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
  bits = np.asarray(bits, dtype=np.int8)
  poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
  payload = bits[:-crc_length]
  expected = crc_encode(payload, crc_length)[-crc_length:]
  return np.array_equal(bits[-crc_length:], expected)


class _Path:
  __slots__ = ("pm", "L", "B", "u_hat")

  def __init__(self, N, n, llr_dec):
    self.pm = 0.0
    self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
    self.B = np.full((N, n + 1), np.nan)
    self.L[:, 0] = llr_dec.copy()
    self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
  """SCL 译码器（Permuted SCD + 路径列表）。"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.frozen_set = set(np.where(self.frozen_bits)[0])
    self.info_indices = np.where(~self.frozen_bits)[0]

  def _pm_penalty(self, llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)

  def _update_llrs(self, path, l):
    for s in range(self.n - _active_llr_level(l, self.n), self.n):
      block_size = 1 << (s + 1)
      branch_size = block_size // 2
      for j in range(l, self.N, block_size):
        if j % block_size < branch_size:
          path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
        else:
          path.L[j, s + 1] = _lower_llr(
            path.L[j, s], path.L[j - branch_size, s], int(path.B[j - branch_size, s + 1])
          )

  def _update_bits(self, path, l):
    if l < self.N // 2:
      return
    for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
      block_size = 1 << s
      branch_size = block_size // 2
      for j in range(l, -1, -block_size):
        if j % block_size >= branch_size:
          path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
          path.B[j, s - 1] = path.B[j, s]

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = self.N
    n = self.n
    rev_perm = np.array([bit_reversed(i, n) for i in range(N)], dtype=int)
    llr_dec = llr_ch[rev_perm]

    paths = [_Path(N, n, llr_dec)]

    for phi in range(N):
      l = bit_reversed(phi, n)
      candidates = []

      for path in paths:
        self._update_llrs(path, l)
        llr = path.L[l, n]

        if l in self.frozen_set:
          new_path = _Path(N, n, llr_dec)
          new_path.pm = path.pm + self._pm_penalty(llr, 0)
          new_path.L = path.L.copy()
          new_path.B = path.B.copy()
          new_path.u_hat = path.u_hat.copy()
          new_path.B[l, n] = 0
          new_path.u_hat[l] = 0
          self._update_bits(new_path, l)
          candidates.append(new_path)
        else:
          for bit in (0, 1):
            new_path = _Path(N, n, llr_dec)
            new_path.pm = path.pm + self._pm_penalty(llr, bit)
            new_path.L = path.L.copy()
            new_path.B = path.B.copy()
            new_path.u_hat = path.u_hat.copy()
            new_path.B[l, n] = bit
            new_path.u_hat[l] = bit
            self._update_bits(new_path, l)
            candidates.append(new_path)

      candidates.sort(key=lambda p: p.pm)
      paths = candidates[: self.list_size]

    if self.crc_length > 0:
      valid = []
      for path in paths:
        info_bits = path.u_hat[self.info_indices]
        if crc_check(info_bits, self.crc_length):
          valid.append(path)
      best = min(valid if valid else paths, key=lambda p: p.pm)
    else:
      best = min(paths, key=lambda p: p.pm)

    return best.u_hat.copy(), best.pm


def verify_scl_equals_sc(N=64, num_frames=50):
  """单路径 SCL 应等价于 SC。"""
  from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
  from construction import ga_construction
  from decoder_sc import sc_decode
  from encoder import polar_encode

  K = N // 2
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=bool)
  frozen_bits[info_idx] = False
  sigma = eb_n0_to_sigma(5.0, K / N)
  rng = np.random.default_rng(1)

  for _ in range(num_frames):
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = rng.integers(0, 2, size=K)
    llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    if not np.array_equal(u_sc, u_scl):
      return False
  return True
