"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import (
  _all_filled,
  _get_up_bit,
  _leftdown,
  _rightdown,
  _up,
  f_operation,
  g_operation,
  sc_decode,
)


def _crc_poly_bits(crc_length):
  if crc_length == 8:
    return [1, 0, 0, 0, 0, 0, 1, 1, 1]
  if crc_length == 16:
    return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后"""
  info_bits = [int(b) for b in info_bits]
  poly = _crc_poly_bits(crc_length)
  crc_n = crc_length
  work = info_bits + [0] * crc_n
  for i in range(len(info_bits)):
    if work[i] == 1:
      for j in range(crc_n + 1):
        work[i + j] ^= poly[j]
  check = work[-crc_n:]
  return np.array(info_bits + check, dtype=np.int8)


def crc_check(bits, crc_length=8):
  bits = [int(b) for b in bits]
  return crc_encode(bits[:-crc_length], crc_length).tolist() == bits


def _init_matrices(llr_ch, n, N):
  llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
  bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
  llr_matrix[0] = llr_ch
  return llr_matrix, bit_matrix


def _get_up_loc(bit_matrix):
  N = bit_matrix.shape[1]
  n = int(math.log2(N))
  detect = -1
  for i in range(N):
    val = bit_matrix[n, i]
    if np.isnan(val):
      detect = i - 1
      break
  if detect == N - 1:
    detect = N - 1
  if detect % 2 == 0:
    loc_row = n - 1
    loc_col = detect
  else:
    loc_row = n - 1
    loc_col = detect - 1
  if detect == -1:
    loc_row = 0
    loc_col = 0
  return [loc_row, loc_col]


def _tree_step(llr_matrix, bit_matrix, frozen_bits, position):
  N = bit_matrix.shape[1]
  span = 2 ** (position[2] - position[0])
  start = position[1]
  up_llr = llr_matrix[position[0]][start : start + span]
  up_bit = bit_matrix[position[0]][start : start + span]
  half = span // 2
  left_llr = llr_matrix[position[0] + 1][start : start + half]
  left_bit = bit_matrix[position[0] + 1][start : start + half]
  right_llr = llr_matrix[position[0] + 1][start + half : start + span]
  right_bit = bit_matrix[position[0] + 1][start + half : start + span]

  if _all_filled(up_bit):
    return _up(position)
  if _all_filled(right_bit):
    bit_matrix[position[0]][start : start + span] = _get_up_bit(left_bit, right_bit)
    return position
  if _all_filled(right_llr):
    if position[0] == position[2] - 1:
      bit_pos = start + half
      decided = 0 if frozen_bits[bit_pos] else (0 if right_llr[0] > 0 else 1)
      bit_matrix[position[0] + 1][start + half : start + span] = decided
      return position
    return _rightdown(position)
  if _all_filled(left_bit):
    llr_matrix[position[0] + 1][start + half : start + span] = np.array(
      [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)]
    )
    return position
  if not _all_filled(left_llr):
    llr_matrix[position[0] + 1][start : start + half] = f_operation(up_llr[:half], up_llr[half:])
    return position
  if position[0] == position[2] - 1:
    bit_pos = start
    decided = 0 if frozen_bits[bit_pos] else (0 if left_llr[0] >= 0 else 1)
    bit_matrix[position[0] + 1][start : start + half] = decided
    return position
  return _leftdown(position)


def _sc_step_to_phi(llr_matrix, bit_matrix, frozen_bits, phi):
  N = bit_matrix.shape[1]
  n = int(math.log2(N))
  if not np.isnan(bit_matrix[n, phi]):
    return llr_matrix, bit_matrix

  loc = _get_up_loc(bit_matrix)
  position = [loc[0], loc[1], n, N]
  guard = 0
  while np.isnan(bit_matrix[n, phi]):
    position = _tree_step(llr_matrix, bit_matrix, frozen_bits, position)
    guard += 1
    if guard > 20 * N * n:
      raise RuntimeError("SC stepping exceeded iteration limit")
  return llr_matrix, bit_matrix


def _pm_update(llr_array, bit_array):
  pm = 0.0
  for llr, bit in zip(llr_array, bit_array):
    hard = 0 if llr >= 0 else 1
    if int(bit) != hard:
      pm += abs(llr)
  return pm


def _finish_decode(llr_matrix, bit_matrix, frozen_bits):
  N = bit_matrix.shape[1]
  n = int(math.log2(N))
  for phi in range(N):
    if np.isnan(bit_matrix[n, phi]):
      _sc_step_to_phi(llr_matrix, bit_matrix, frozen_bits, phi)
  return bit_matrix[n].astype(int)


class SCLDecoder:
  """SCL 译码器"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.info_indices = np.where(~self.frozen_bits)[0].tolist()

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = self.N
    n = self.n

    if self.list_size == 1 and self.crc_length == 0:
      return sc_decode(llr_ch, self.frozen_bits), 0.0

    split_pos = self.info_indices
    llr_list = [_init_matrices(llr_ch, n, N)[0]]
    bit_list = [_init_matrices(llr_ch, n, N)[1]]
    pm_list = [0.0]
    split_loc = 0

    while split_loc < len(split_pos):
      l_now = len(pm_list)
      new_llr, new_bit, new_pm = [], [], []
      prev = split_pos[split_loc - 1] if split_loc > 0 else -1
      cur = split_pos[split_loc]

      for i in range(l_now):
        lm = llr_list[i].copy()
        bm = bit_list[i].copy()
        pm = pm_list[i]
        lm, bm = _sc_step_to_phi(lm, bm, self.frozen_bits, cur)

        seg_llr = lm[n][prev + 1 : cur + 1]
        seg_bit = bm[n][prev + 1 : cur + 1].astype(int)

        for bit_val in (0, 1):
          lm_copy = lm.copy()
          bm_copy = bm.copy()
          bm_copy[n, cur] = bit_val
          seg_bit_copy = seg_bit.copy()
          seg_bit_copy[-1] = bit_val
          new_pm.append(pm + _pm_update(seg_llr, seg_bit_copy))
          new_llr.append(lm_copy)
          new_bit.append(bm_copy)

      order = np.argsort(new_pm)
      keep = order[: self.list_size]
      llr_list = [new_llr[i] for i in keep]
      bit_list = [new_bit[i] for i in keep]
      pm_list = [new_pm[i] for i in keep]
      split_loc += 1

    if split_pos and split_pos[-1] != N - 1:
      l_now = len(pm_list)
      prev = split_pos[-1]
      for i in range(l_now):
        lm, bm = _sc_step_to_phi(llr_list[i].copy(), bit_list[i].copy(), self.frozen_bits, N - 1)
        seg_llr = lm[n][prev + 1 : N]
        seg_bit = bm[n][prev + 1 : N].astype(int)
        pm_list[i] += _pm_update(seg_llr, seg_bit)
        llr_list[i] = lm
        bit_list[i] = bm

    order = np.argsort(pm_list)
    candidates = []
    for idx in order:
      u_hat = _finish_decode(llr_list[idx].copy(), bit_list[idx].copy(), self.frozen_bits)
      candidates.append((pm_list[idx], u_hat))

    if self.crc_length > 0:
      for pm, u_hat in candidates:
        if crc_check(u_hat[self.info_indices], self.crc_length):
          return u_hat, pm

    return candidates[0][1], candidates[0][0]
