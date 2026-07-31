"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
  _active_bit_level,
  _active_llr_level,
  f_operation,
  lower_llr,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, crc_length=8):
  """计算 CRC 余数（MSB first LFSR）。"""
  poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
  crc = 0
  mask = (1 << crc_length) - 1
  for bit in bits:
    feedback = ((crc >> (crc_length - 1)) ^ int(bit)) & 1
    crc = ((crc << 1) & mask) ^ (feedback * poly)
  return crc


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后。"""
  info_bits = np.asarray(info_bits, dtype=int)
  rem = _crc_remainder(info_bits, crc_length)
  crc_bits = np.array(
    [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
    dtype=int,
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits 是否通过 CRC。"""
  return _crc_remainder(bits, crc_length) == 0


class Path:
  """单条 SCL 路径。"""

  __slots__ = ("pm", "B", "L")

  def __init__(self, n, N):
    self.pm = 0.0
    self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
    self.B = np.zeros((N, n + 1))


class SCLDecoder:
  """SCL 译码器（PSCD 风格，Lazy Copy）。"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.frozen_set = set(np.where(self.frozen_bits)[0])
    self.list_size = list_size
    self.crc_length = crc_length
    self.br = bit_reversal_permutation(N)

  def _copy_path(self, path):
    new_path = Path(self.n, self.N)
    new_path.pm = path.pm
    new_path.L = path.L.copy()
    new_path.B = path.B.copy()
    return new_path

  def _update_llrs(self, path, l):
    for s in range(self.n - _active_llr_level(l, self.n), self.n):
      block_size = 2 ** (s + 1)
      branch_size = block_size // 2
      for j in range(l, self.N, block_size):
        if j % block_size < branch_size:
          path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
        else:
          path.L[j, s + 1] = lower_llr(
            path.L[j, s],
            path.L[j - branch_size, s],
            path.B[j - branch_size, s + 1],
          )

  def _propagate_bits(self, path, l):
    if l < self.N / 2:
      return
    for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
      block_size = 2 ** s
      branch_size = block_size // 2
      for j in range(l, -1, -block_size):
        if j % block_size >= branch_size:
          path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
            path.B[j - branch_size, s]
          )
          path.B[j, s - 1] = path.B[j, s]

  def _path_metric_penalty(self, llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)

  def decode(self, llr_ch):
    """主译码函数，返回 (u_hat, pm)。"""
    llr_perm = llr_ch[self.br].astype(np.float64)

    paths = [Path(self.n, self.N)]
    paths[0].L[:, 0] = llr_perm

    for phi in range(self.N):
      l = self.br[phi]
      candidates = []

      for path in paths:
        self._update_llrs(path, l)
        llr = path.L[l, self.n]

        if l in self.frozen_set:
          bit = 0
          new_path = self._copy_path(path)
          new_path.pm += self._path_metric_penalty(llr, bit)
          new_path.B[l, self.n] = bit
          self._propagate_bits(new_path, l)
          candidates.append(new_path)
        else:
          for bit in (0, 1):
            new_path = self._copy_path(path)
            new_path.pm += self._path_metric_penalty(llr, bit)
            new_path.B[l, self.n] = bit
            self._propagate_bits(new_path, l)
            candidates.append(new_path)

      candidates.sort(key=lambda p: p.pm)
      paths = candidates[: self.list_size]

    crc_valid = []
    for path in paths:
      u_hat = path.B[:, self.n].astype(int)
      if self.crc_length > 0:
        info_bits = u_hat[~self.frozen_bits]
        if crc_check(info_bits, self.crc_length):
          crc_valid.append(path)
      else:
        crc_valid.append(path)

    pool = crc_valid if crc_valid else paths
    best = min(pool, key=lambda p: p.pm)
    return best.B[:, self.n].astype(int).copy(), best.pm
