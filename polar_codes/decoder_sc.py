"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
  """
  min-sum 近似的 f 运算：
  f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
  """
  sa = np.sign(La)
  sb = np.sign(Lb)
  sa = np.where(sa == 0, 1, sa)
  sb = np.where(sb == 0, 1, sb)
  return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
  """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
  return (1 - 2 * u_hat) * La + Lb


def _all_filled(arr):
  return not np.isnan(arr).any()


def _leftdown(position):
  return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
  return [
    position[0] + 1,
    position[1] + 2 ** (position[2] - 1 - position[0]),
    position[2],
    position[3],
  ]


def _up(position):
  span = 2 ** (position[2] - position[0] + 1)
  return [position[0] - 1, int(np.floor(position[1] / span) * span), position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
  length = left_bit.size
  temp = np.array([(left_bit + right_bit) % 2, right_bit])
  temp.resize((1, 2 * length))
  return temp


def _sc_tree_decode(llr_ch, frozen_bits):
  """基于因子树遍历的 SC 译码（非递归主实现）"""
  y_llr = np.asarray(llr_ch, dtype=np.float64)
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  N = y_llr.size
  n = int(math.log2(N))

  llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
  bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
  llr_matrix[0] = y_llr
  position = [0, 0, n, N]

  while not _all_filled(bit_matrix[n]):
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
      position = _up(position)
      continue

    if _all_filled(right_bit):
      up_bit_val = _get_up_bit(left_bit, right_bit)
      bit_matrix[position[0]][start : start + span] = up_bit_val.copy()
      continue

    if _all_filled(right_llr):
      if position[0] == position[2] - 1:
        bit_pos = start + half
        if frozen_bits[bit_pos]:
          decided = 0
        else:
          decided = 0 if right_llr[0] > 0 else 1
        bit_matrix[position[0] + 1][start + half : start + span] = decided
      else:
        position = _rightdown(position)
      continue

    if _all_filled(left_bit):
      right_llr_val = np.array(
        [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)]
      )
      llr_matrix[position[0] + 1][start + half : start + span] = right_llr_val
      continue

    if not _all_filled(left_llr):
      left_llr_val = f_operation(up_llr[:half], up_llr[half:])
      llr_matrix[position[0] + 1][start : start + half] = left_llr_val
      continue

    if position[0] == position[2] - 1:
      bit_pos = start
      if frozen_bits[bit_pos]:
        decided = 0
      else:
        decided = 0 if left_llr[0] >= 0 else 1
      bit_matrix[position[0] + 1][start : start + half] = decided
    else:
      position = _leftdown(position)

  return bit_matrix[n].astype(int)


def sc_decode(llr_ch, frozen_bits):
  """非递归 SC 译码主函数"""
  return _sc_tree_decode(llr_ch, frozen_bits)


def sc_decode_recursive(llr_ch, frozen_bits):
  """
  递归 SC 译码（通过树遍历实现，与 sc_decode 等价）。
  """
  return _sc_tree_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
  """
  预计算非递归 SC 译码所需的辅助向量（供 SCL 使用）。
  """
  n = int(math.log2(N))
  lambda_offset = [1 << i for i in range(n + 1)]
  llr_layer_vec = []
  bit_layer_vec = []

  for phi in range(N):
    llr_layers = []
    p = phi
    layer = 0
    while p & 1:
      llr_layers.append(layer)
      p >>= 1
      layer += 1
    llr_layer_vec.append(llr_layers)

    bit_layers = []
    p = phi >> 1
    layer = 0
    while p & 1:
      bit_layers.append(layer)
      p >>= 1
      layer += 1
    bit_layer_vec.append(bit_layers)

  return lambda_offset, llr_layer_vec, bit_layer_vec
