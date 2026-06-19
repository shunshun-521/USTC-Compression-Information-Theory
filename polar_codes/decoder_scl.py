"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _all_num,
    _channel_llr_for_decode,
    _get_left_bit,
    _get_left_llr,
    _get_right_bit,
    _get_right_llr,
    _get_up_bit,
    _leftdown,
    _rightdown,
    _up,
    sc_decode_tree,
)


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后。"""
  info_bits = np.asarray(info_bits, dtype=int)
  if crc_length == 8:
    poly = 0x07
  elif crc_length == 16:
    poly = 0x8005
  else:
    raise ValueError("crc_length must be 8 or 16")

  reg = 0
  for bit in info_bits:
    reg ^= int(bit) << (crc_length - 1)
    for _ in range(8):
      if reg & (1 << (crc_length - 1)):
        reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
      else:
        reg = (reg << 1) & ((1 << crc_length) - 1)

  crc_bits = np.array(
      [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits 末尾 CRC 是否正确。"""
  bits = np.asarray(bits, dtype=int)
  return np.array_equal(
      crc_encode(bits[:-crc_length], crc_length)[-crc_length:],
      bits[-crc_length:],
  )


def _get_up_loc(bit_matrix):
  N = bit_matrix.shape[1]
  n = int(math.log2(N))
  detect_array = bit_matrix[n]
  detect = -1
  for i in range(N):
    if detect_array[i] == 0 or detect_array[i] == 1:
      continue
    detect = i - 1
    break
  if detect % 2 == 0:
    loc_row = n - 1
    loc_col = detect
  else:
    loc_row = n - 1
    loc_col = detect - 1
  if detect == -1:
    loc_row, loc_col = 0, 0
  return [loc_row, loc_col]


def _sc_step_until(llr_matrix, bit_matrix, frozen_bits, split_pos):
  """SC 树遍历直至比特 split_pos 判决完成。"""
  N = bit_matrix.shape[1]
  n = int(math.log2(N))
  loc = _get_up_loc(bit_matrix)
  position = [loc[0], loc[1], n, N]

  while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
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

    if _all_num(up_bit):
      position = _up(position)
    elif _all_num(right_bit):
      up_bit_val = _get_up_bit(left_bit, right_bit)
      bit_matrix[position[0]][
          position[1]:position[1] + 2 ** (position[2] - position[0])
      ] = up_bit_val.copy()
    elif _all_num(right_llr):
      if position[0] == position[2] - 1:
        right_bit_pos = position[1] + 1
        right_bit_val = _get_right_bit(right_llr[0], frozen_bits, right_bit_pos)
        bit_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1):
            position[1] + 2 ** (position[2] - position[0])
        ] = right_bit_val
      else:
        position = _rightdown(position)
    elif _all_num(left_bit):
      right_llr_val = _get_right_llr(left_bit, up_llr)
      llr_matrix[position[0] + 1][
          position[1] + 2 ** (position[2] - position[0] - 1):
          position[1] + 2 ** (position[2] - position[0])
      ] = right_llr_val
    elif not _all_num(left_llr):
      left_llr_val = _get_left_llr(up_llr)
      llr_matrix[position[0] + 1][
          position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
      ] = left_llr_val
    elif position[0] == position[2] - 1:
      left_bit_pos = position[1]
      left_bit_val = _get_left_bit(left_llr[0], frozen_bits, left_bit_pos)
      bit_matrix[position[0] + 1][
          position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
      ] = left_bit_val
    else:
      position = _leftdown(position)

  return llr_matrix, bit_matrix


def _pm_update(llr_val, bit_val, pm):
  hard = 0 if llr_val >= 0 else 1
  if bit_val != hard:
    pm += abs(llr_val)
  return pm


class SCLDecoder:
  """SCL 译码器。"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.info_indices = np.where(~self.frozen_bits)[0]

  def decode(self, llr_ch):
    if self.list_size == 1:
      u_hat = sc_decode_tree(llr_ch, self.frozen_bits)
      return u_hat, 0.0

    llr_ch = _channel_llr_for_decode(llr_ch)
    N, n = self.N, self.n

    llr_paths = [np.full((n + 1, N), np.nan)]
    bit_paths = [np.full((n + 1, N), np.nan)]
    llr_paths[0][0] = llr_ch
    pm_list = [0.0]

    for phi in range(N):
      new_llr, new_bit, new_pm = [], [], []

      for pidx, (lm, bm) in enumerate(zip(llr_paths, bit_paths)):
        lm_c = lm.copy()
        bm_c = bm.copy()
        lm_c, bm_c = _sc_step_until(lm_c, bm_c, self.frozen_bits, phi)
        leaf_llr = lm_c[n, phi]

        if self.frozen_bits[phi]:
          pm_new = _pm_update(leaf_llr, 0, pm_list[pidx])
          new_llr.append(lm_c)
          new_bit.append(bm_c)
          new_pm.append(pm_new)
        else:
          for bit in (0, 1):
            lm_b = lm_c.copy()
            bm_b = bm_c.copy()
            bm_b[n, phi] = bit
            pm_new = _pm_update(leaf_llr, bit, pm_list[pidx])
            new_llr.append(lm_b)
            new_bit.append(bm_b)
            new_pm.append(pm_new)

      order = np.argsort(new_pm)[: self.list_size]
      llr_paths = [new_llr[i] for i in order]
      bit_paths = [new_bit[i] for i in order]
      pm_list = [new_pm[i] for i in order]

    candidates = [(pm_list[i], bit_paths[i][n].astype(int)) for i in range(len(pm_list))]

    if self.crc_length > 0:
      for pm, u_hat in sorted(candidates, key=lambda x: x[0]):
        info_bits = u_hat[self.info_indices]
        if crc_check(info_bits, self.crc_length):
          return u_hat, pm

    best = min(candidates, key=lambda x: x[0])
    return best[1], best[0]
