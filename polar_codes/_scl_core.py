"""SCL 译码器核心（改编自参考实现）。"""
import math

import numpy as np

from _ref_function import (
  all_num,
  get_left_bit,
  get_left_llr,
  get_right_bit,
  get_right_llr,
  get_up_bit,
  get_up_loc,
  leftdown,
  rightdown,
  up,
)


def get_pm_update(llr_array, bit_array):
  pm = 0.0
  for i in range(llr_array.size):
    if np.sign(llr_array[i]) != np.sign(1 - 2 * bit_array[i]):
      pm += np.abs(llr_array[i])
  return pm


def sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
  N = int(bit_matrix[0].size)
  n = int(math.log2(N))
  loc = get_up_loc(bit_matrix)
  position = [loc[0], loc[1], n, N]

  while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
    up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
    up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
    left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
    left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
    right_llr = llr_matrix[position[0] + 1][
      position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
    ]
    right_bit = bit_matrix[position[0] + 1][
      position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
    ]

    if all_num(up_bit) == 1:
      position = up(position)
    else:
      if all_num(right_bit) == 1:
        up_b = get_up_bit(left_bit, right_bit)
        bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_b.copy()
      else:
        if all_num(right_llr) == 1:
          if position[0] == position[2] - 1:
            right_bit_pos = position[1] + 1
            rb = get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
            bit_matrix[position[0] + 1][
              position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
            ] = rb
          else:
            position = rightdown(position)
        else:
          if all_num(left_bit) == 1:
            right_l = get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][
              position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])
            ] = right_l
          else:
            if all_num(left_llr) == 0:
              left_l = get_left_llr(up_llr)
              llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_l
            else:
              if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                lb = get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
                bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = lb
              else:
                position = leftdown(position)

  return llr_matrix, bit_matrix


def scl_tree_decode(llr_ch, information_pos, list_size, crc_check_fn=None):
  """SCL 译码。"""
  N = llr_ch.size
  n = int(math.log2(N))
  information_pos = list(information_pos)
  frozen_bit = 0
  split_pos = information_pos

  llr_matrix = np.ones((n + 1, N))
  llr_matrix[llr_matrix == 1] = float("nan")
  bit_matrix = llr_matrix.copy()
  llr_matrix[0] = llr_ch

  llr_list = [llr_matrix]
  bit_list = [bit_matrix]
  pm_list = [0.0]
  split_loc = 0
  split_len = len(split_pos)
  l_now = 1

  while split_len - 1 >= split_loc:
    new_llr, new_bit, new_pm = [], [], []
    for i in range(l_now):
      llr_temp = llr_list[i].copy()
      bit_temp = bit_list[i].copy()
      pm_temp = pm_list[i]

      llr_out, bit_out = sc_stepping_decoder(
        llr_temp, bit_temp, information_pos, frozen_bit, split_pos[split_loc]
      )
      prev = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
      curr = split_pos[split_loc] + 1
      pm_slice = get_pm_update(llr_out[n][prev:curr], bit_out[n][prev:curr])

      new_llr.append(llr_out)
      new_bit.append(bit_out)
      new_pm.append(pm_temp + pm_slice)

      llr_wrong = llr_out.copy()
      bit_wrong = bit_out.copy()
      bit_wrong[n][split_pos[split_loc]] = 1 - bit_wrong[n][split_pos[split_loc]]
      pm_wrong = get_pm_update(llr_out[n][prev:curr], bit_wrong[n][prev:curr])
      new_llr.append(llr_wrong)
      new_bit.append(bit_wrong)
      new_pm.append(pm_temp + pm_wrong)

    if l_now > list_size / 2:
      order = np.argsort(new_pm)[:list_size]
      llr_list = [new_llr[i] for i in order]
      bit_list = [new_bit[i] for i in order]
      pm_list = [new_pm[i] for i in order]
    else:
      llr_list, bit_list, pm_list = new_llr, new_bit, new_pm
    l_now = len(pm_list)
    split_loc += 1

  if split_pos[-1] != N - 1:
    for i in range(l_now):
      llr_temp = llr_list[i].copy()
      bit_temp = bit_list[i].copy()
      pm_temp = pm_list[i]
      llr_out, bit_out = sc_stepping_decoder(
        llr_temp, bit_temp, information_pos, frozen_bit, N - 1
      )
      prev = split_pos[split_loc - 1] + 1
      pm_slice = get_pm_update(llr_out[n][prev:N], bit_out[n][prev:N])
      llr_list[i] = llr_out
      bit_list[i] = bit_out
      pm_list[i] = pm_temp + pm_slice

  order = np.argsort(pm_list)
  best_u = None
  best_pm = pm_list[order[0]]
  for idx in order:
    u_d = np.array([0 if bit_list[idx][n][i] == 0 else 1 for i in range(N)])
    if crc_check_fn is None or crc_check_fn(u_d[information_pos]):
      best_u = u_d
      best_pm = pm_list[idx]
      break
  if best_u is None:
    idx = order[0]
    best_u = np.array([0 if bit_list[idx][n][i] == 0 else 1 for i in range(N)])
    best_pm = pm_list[idx]

  return best_u, best_pm
