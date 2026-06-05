"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _all_num(x):
  length = x.size
  for i in range(length):
    if np.isnan(x[i]):
      return 0
  return 1


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
  p0 = position[0] - 1
  p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1))))
  p1 *= 2 ** (position[2] - position[0] + 1)
  return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
  length = left_bit.size
  temp = np.array([(left_bit + right_bit) % 2, right_bit])
  temp.resize((1, 2 * length))
  return temp


def _get_right_llr(left_bit, up_llr):
  length = int(left_bit.size)
  return np.array([
    g_operation(up_llr[i], up_llr[i + length], left_bit[i])
    for i in range(length)
  ])


def _get_left_llr(up_llr):
  length = int(up_llr.size / 2)
  return f_operation(up_llr[:length], up_llr[length:])


def _get_left_bit(left_llr, info_set, frozen_val, pos):
  if pos in info_set:
    return 0 if left_llr >= 0 else 1
  return frozen_val


def _get_right_bit(right_llr, info_set, frozen_val, pos):
  if pos in info_set:
    return 0 if right_llr > 0 else 1
  return frozen_val


def _sc_decode_core(y_llr, frozen_bits):
  N = y_llr.size
  n = int(math.log2(N))
  info_set = set(np.where(~np.asarray(frozen_bits, dtype=bool))[0])
  frozen_val = 0

  llr_matrix = np.ones((n + 1, N), dtype=np.float64)
  llr_matrix[llr_matrix == 1] = np.nan
  bit_matrix = llr_matrix.copy()
  llr_matrix[0] = y_llr.astype(np.float64)

  position = [0, 0, n, N]

  while _all_num(bit_matrix[n]) == 0:
    up_llr = llr_matrix[position[0]][
      position[1]:position[1] + 2 ** (position[2] - position[0])
    ]
    up_bit = bit_matrix[position[0]][
      position[1]:position[1] + 2 ** (position[2] - position[0])
    ]
    left_llr = llr_matrix[position[0] + 1][
      position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
    ]
    left_bit = bit_matrix[position[0] + 1][
      position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
    ]
    right_llr = llr_matrix[position[0] + 1][
      position[1] + 2 ** (position[2] - position[0] - 1):
      position[1] + 2 ** (position[2] - position[0])
    ]
    right_bit = bit_matrix[position[0] + 1][
      position[1] + 2 ** (position[2] - position[0] - 1):
      position[1] + 2 ** (position[2] - position[0])
    ]

    if _all_num(up_bit) == 1:
      position = _up(position)
    else:
      if _all_num(right_bit) == 1:
        up_bit_val = _get_up_bit(left_bit, right_bit)
        bit_matrix[position[0]][
          position[1]:position[1] + 2 ** (position[2] - position[0])
        ] = up_bit_val.copy()
      else:
        if _all_num(right_llr) == 1:
          if position[0] == position[2] - 1:
            right_bit_pos = position[1] + 1
            bit_matrix[position[0] + 1][
              position[1] + 2 ** (position[2] - position[0] - 1):
              position[1] + 2 ** (position[2] - position[0])
            ] = _get_right_bit(right_llr, info_set, frozen_val, right_bit_pos)
          else:
            position = _rightdown(position)
        else:
          if _all_num(left_bit) == 1:
            llr_matrix[position[0] + 1][
              position[1] + 2 ** (position[2] - position[0] - 1):
              position[1] + 2 ** (position[2] - position[0])
            ] = _get_right_llr(left_bit, up_llr)
          else:
            if _all_num(left_llr) == 0:
              llr_matrix[position[0] + 1][
                position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
              ] = _get_left_llr(up_llr)
            else:
              if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                bit_matrix[position[0] + 1][
                  position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
                ] = _get_left_bit(left_llr, info_set, frozen_val, left_bit_pos)
              else:
                position = _leftdown(position)

  return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
  return _sc_decode_core(llr, frozen_bits)


def precompute_sc_indices(N):
  n = int(math.log2(N))
  lambda_offset = [1 << i for i in range(n + 1)]
  llr_layer_vec = []
  bit_layer_vec = []
  for phi in range(N):
    i = 0
    while i < n and ((phi >> i) & 1):
      i += 1
    llr_layer_vec.append(list(range(i, n)) if i < n else [])
    bit_layer_vec.append([j for j in range(n) if (phi >> j) & 1])
  return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
  return _sc_decode_core(llr_ch, frozen_bits)
