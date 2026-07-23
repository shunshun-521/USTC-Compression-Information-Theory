"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    lower_llr,
    sc_decode,
    upper_llr,
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
  crc_bits = np.array(
      [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
      dtype=np.int8,
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
  expected = crc_encode(bits[:-crc_length], crc_length)
  return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _path_metric_update(pm, llr, u):
  u_from_llr = 0 if llr >= 0 else 1
  if u != u_from_llr:
    pm += abs(llr)
  return pm


class _PathState:
  __slots__ = ("L", "B", "pm")

  def __init__(self, N, n, llr_ch):
    self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=int)
    self.L[:, 0] = llr_ch
    self.pm = 0.0

  def copy(self):
    p = _PathState.__new__(_PathState)
    p.L = self.L.copy()
    p.B = self.B.copy()
    p.pm = self.pm
    return p


class SCLDecoder:
  """SCL 译码器（Permuted SCD + 路径度量）"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.info_indices = np.where(~self.frozen_bits)[0]

  def _update_llrs(self, path, l):
    L, B = path.L, path.B
    n = self.n
    N = self.N
    for s in range(n - _active_llr_level(l, n), n):
      block_size = 2 ** (s + 1)
      branch_size = block_size // 2
      for j in range(l, N, block_size):
        if j % block_size < branch_size:
          L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
        else:
          L[j, s + 1] = lower_llr(
              L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
          )

  def _update_bits(self, path, l):
    if l < self.N // 2:
      return
    L, B = path.L, path.B
    n = self.n
    N = self.N
    for s in range(n, n - _active_bit_level(l, n), -1):
      block_size = 2 ** s
      branch_size = block_size // 2
      for j in range(l, -1, -block_size):
        if j % block_size >= branch_size:
          B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
          B[j, s - 1] = B[j, s]

  def decode(self, llr_ch):
    if self.list_size == 1 and self.crc_length == 0:
      return sc_decode(llr_ch, self.frozen_bits), 0.0

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    n, N = self.n, self.N
    paths = [_PathState(N, n, llr_ch)]

    for l in [_bit_reversed(i, n) for i in range(N)]:
      is_frozen = self.frozen_bits[l]
      candidates = []

      for path in paths:
        self._update_llrs(path, l)
        llr = path.L[l, n]
        if is_frozen:
          new_path = path.copy()
          new_path.pm = _path_metric_update(path.pm, llr, 0)
          new_path.B[l, n] = 0
          self._update_bits(new_path, l)
          candidates.append(new_path)
        else:
          for u in (0, 1):
            new_path = path.copy()
            new_path.pm = _path_metric_update(path.pm, llr, u)
            new_path.B[l, n] = u
            self._update_bits(new_path, l)
            candidates.append(new_path)

      candidates.sort(key=lambda p: p.pm)
      paths = candidates[: self.list_size]

    if self.crc_length > 0:
      valid = []
      for p in paths:
        u_hat = p.B[:, n].astype(int)
        if self._crc_pass(u_hat):
          valid.append(p)
      if valid:
        paths = valid

    best = min(paths, key=lambda p: p.pm)
    return best.B[:, n].astype(int), best.pm

  def _crc_pass(self, u_hat):
    info_bits = u_hat[self.info_indices]
    if len(info_bits) < self.crc_length:
      return False
    return crc_check(info_bits, self.crc_length)
