"""SC 译码器内部辅助函数（因子树遍历，改编自参考实现）。"""
from _ref_function import (
  all_num,
  get_left_bit,
  get_left_llr,
  get_right_bit,
  get_right_llr,
  get_up_bit,
  leftdown,
  rightdown,
  up,
)
import math

import numpy as np


def sc_tree_decode(llr_ch, information_pos, frozen_bit=0):
  """因子树遍历 SC 译码。"""
  N = llr_ch.size
  n = int(math.log2(N))
  information_pos = list(information_pos)

  llr_matrix = np.ones((n + 1, N))
  llr_matrix[llr_matrix == 1] = float("nan")
  bit_matrix = llr_matrix.copy()
  llr_matrix[0] = llr_ch
  position = [0, 0, n, N]

  while all_num(bit_matrix[n]) == 0:
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

  return np.array([0 if bit_matrix[n][i] == 0 else 1 for i in range(N)])
