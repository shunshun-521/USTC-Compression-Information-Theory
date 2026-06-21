"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, sc_decode

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
  mask = (1 << crc_length) - 1
  reg = 0
  for b in bits:
    reg ^= int(b) << (crc_length - 1)
    if reg & (1 << (crc_length - 1)):
      reg = ((reg << 1) ^ poly) & mask
    else:
      reg = (reg << 1) & mask
  return reg


def crc_encode(info_bits, crc_length=8):
  info_bits = np.asarray(info_bits, dtype=int)
  if crc_length == 8:
    poly = _CRC8_POLY
  elif crc_length == 16:
    poly = _CRC16_POLY
  else:
    raise ValueError("crc_length 仅支持 8 或 16")
  rem = _crc_remainder(info_bits, poly, crc_length)
  crc_bits = np.array(
    [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  bits = np.asarray(bits, dtype=int)
  if crc_length == 8:
    poly = _CRC8_POLY
  elif crc_length == 16:
    poly = _CRC16_POLY
  else:
    raise ValueError("crc_length 仅支持 8 或 16")
  return _crc_remainder(bits, poly, crc_length) == 0


def _path_metric_update(pm, llr, bit):
  hard = 0 if llr >= 0 else 1
  if bit != hard:
    pm += abs(llr)
  return pm


def _scl_recursive(llr_node, frozen_node, list_size):
  """递归 SCL，返回 [(pm, u_hat), ...]"""
  n = len(llr_node)
  if n == 1:
    llr = llr_node[0]
    if frozen_node[0]:
      u = np.array([0], dtype=int)
      return [(_path_metric_update(0.0, llr, 0), u)]
    cands = []
    for b in (0, 1):
      u = np.array([b], dtype=int)
      cands.append((_path_metric_update(0.0, llr, b), u))
    cands.sort(key=lambda x: x[0])
    return cands[:list_size]

  half = n // 2
  llr1 = llr_node[:half]
  llr2 = llr_node[half:]
  f1 = frozen_node[:half]
  f2 = frozen_node[half:]

  llr_left = f_operation(llr1, llr2)
  left_paths = _scl_recursive(llr_left, f1, list_size)

  merged = []
  for pm_l, u_l in left_paths:
    u_l_up = _compute_u_up(u_l)
    llr_right = g_operation(llr1, llr2, u_l_up)
    right_paths = _scl_recursive(llr_right, f2, list_size)
    for pm_r, u_r in right_paths:
      merged.append((pm_l + pm_r, np.concatenate([u_l, u_r])))

  merged.sort(key=lambda x: x[0])
  return merged[:list_size]


def _compute_u_up(u_hat):
  """由已译码比特计算 g 运算所需的部分和（与 SC 递归一致）"""
  n = len(u_hat)
  if n == 1:
    return u_hat.astype(int)
  half = n // 2
  u_l_up = _compute_u_up(u_hat[:half])
  u_r_up = _compute_u_up(u_hat[half:])
  merged = (u_l_up.astype(int) ^ u_r_up.astype(int)).astype(int)
  return np.concatenate([merged, u_r_up])


class SCLDecoder:
  """SCL 译码器"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.br = bit_reversal_permutation(N)
    self.info_indices = np.where(~self.frozen_bits)[0]

  def decode(self, llr_ch):
    if self.list_size == 1:
      u_hat = sc_decode(llr_ch, self.frozen_bits)
      return u_hat, 0.0

    llr = np.asarray(llr_ch, dtype=np.float64)[self.br]
    paths = _scl_recursive(llr, self.frozen_bits, self.list_size)

    if self.crc_length > 0:
      valid = [(pm, u) for pm, u in paths
               if crc_check(u[self.info_indices], self.crc_length)]
      if valid:
        paths = valid

    best_pm, best_u = min(paths, key=lambda x: x[0])
    return best_u.astype(int), best_pm
