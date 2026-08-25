"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
  _active_bit_level,
  _active_llr_level,
  _bit_reversed,
  _lower_llr,
  _upper_llr,
)


CRC_POLYNOMIALS = {
  8: 0x07,
  16: 0x8005,
}


def _crc_remainder(bits, crc_length):
  poly = CRC_POLYNOMIALS[crc_length]
  reg = 0
  for bit in np.asarray(bits, dtype=np.int8):
    reg ^= int(bit) << (crc_length - 1)
    if reg & (1 << (crc_length - 1)):
      reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
    else:
      reg = (reg << 1) & ((1 << crc_length) - 1)
  return reg


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后"""
  info_bits = np.asarray(info_bits, dtype=np.int8)
  remainder = _crc_remainder(info_bits, crc_length)
  crc_bits = np.array(
    [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
    dtype=np.int8,
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits 是否通过 CRC 校验"""
  return _crc_remainder(bits, crc_length) == 0


class Path:
  """单条 SCL 路径"""

  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n):
    self.L = np.zeros((N, n + 1), dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=np.int8)
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=np.int8)

  def copy(self):
    p = Path(self.L.shape[0], self.L.shape[1] - 1)
    p.L = self.L.copy()
    p.B = self.B.copy()
    p.pm = self.pm
    p.u_hat = self.u_hat.copy()
    return p


class SCLDecoder:
  """SCL 译码器（Permuted SC + Lazy Copy）"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.frozen_set = set(np.where(self.frozen_bits)[0])
    self.list_size = list_size
    self.crc_length = crc_length
    self.bit_rev = bit_reversal_permutation(N)

  def _update_llrs(self, path, l):
    for s in range(self.n - _active_llr_level(l, self.n), self.n):
      block_size = 2 ** (s + 1)
      branch_size = block_size // 2
      for j in range(l, self.N, block_size):
        if j % block_size < branch_size:
          path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
        else:
          top_bit = path.B[j - branch_size, s + 1]
          path.L[j, s + 1] = _lower_llr(
            path.L[j, s], path.L[j - branch_size, s], top_bit
          )

  def _update_bits(self, path, l):
    if l < self.N // 2:
      return
    for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
      block_size = 2 ** s
      branch_size = block_size // 2
      for j in range(l, -1, -block_size):
        if j % block_size >= branch_size:
          path.B[j - branch_size, s - 1] = (
            path.B[j, s] ^ path.B[j - branch_size, s]
          ) & 1
          path.B[j, s - 1] = path.B[j, s]

  def _penalty(self, llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)

  def decode(self, llr_ch):
    llr = np.asarray(llr_ch, dtype=np.float64)[self.bit_rev]
    paths = [Path(self.N, self.n)]
    paths[0].L[:, 0] = llr

    for phi in range(self.N):
      l = _bit_reversed(phi, self.n)
      new_paths = []

      for path in paths:
        self._update_llrs(path, l)
        cur_llr = path.L[l, self.n]

        if l in self.frozen_set:
          path.pm += self._penalty(cur_llr, 0)
          path.B[l, self.n] = 0
          path.u_hat[l] = 0
          self._update_bits(path, l)
          new_paths.append(path)
        else:
          for bit in (0, 1):
            child = path.copy()
            child.pm += self._penalty(cur_llr, bit)
            child.B[l, self.n] = bit
            child.u_hat[l] = bit
            self._update_bits(child, l)
            new_paths.append(child)

      new_paths.sort(key=lambda p: p.pm)
      paths = new_paths[: self.list_size]

    if self.crc_length > 0:
      info_positions = np.where(~self.frozen_bits)[0]
      valid = [
        p
        for p in paths
        if crc_check(p.u_hat[info_positions], self.crc_length)
      ]
      best = min(valid if valid else paths, key=lambda p: p.pm)
    else:
      best = min(paths, key=lambda p: p.pm)

    return best.u_hat.copy(), best.pm
