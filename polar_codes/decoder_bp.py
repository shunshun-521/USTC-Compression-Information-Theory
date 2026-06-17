"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import build_generator_matrix, polar_encode


def _f_hf_sms(l1, l2, alpha=0.9375):
  s1 = np.sign(l1)
  s2 = np.sign(l2)
  if s1 == 0:
    s1 = 1
  if s2 == 0:
    s2 = 1
  return alpha * s1 * s2 * min(abs(l1), abs(l2))


def _element_update_left(left, right, alpha):
  value = np.zeros(2)
  value[0] = _f_hf_sms(right[1] + left[1], left[0], alpha)
  value[1] = _f_hf_sms(left[0], right[0], alpha) + left[1]
  return value


def _element_update_right(left, right, alpha):
  value = np.zeros(2)
  value[0] = _f_hf_sms(right[1] + left[1], right[0], alpha)
  value[1] = _f_hf_sms(left[0], right[0], alpha) + right[1]
  return value


def _bp_update_left(left_array, right_array, stage, alpha):
  N = left_array.size
  interval = 2 ** (stage - 1)
  num = N // (interval * 2)
  value = np.zeros(N)
  for i in range(num):
    for j in range(interval):
      left_ele = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
      right_ele = np.array([right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]])
      get_value = _element_update_left(left_ele, right_ele, alpha)
      value[2 * i * interval + j] = get_value[0]
      value[2 * i * interval + j + interval] = get_value[1]
  return value


def _bp_update_right(left_array, right_array, stage, alpha):
  N = left_array.size
  interval = 2 ** (stage - 1)
  num = N // (interval * 2)
  value = np.zeros(N)
  for i in range(num):
    for j in range(interval):
      left_ele = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
      right_ele = np.array([right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]])
      get_value = _element_update_right(left_ele, right_ele, alpha)
      value[2 * i * interval + j] = get_value[0]
      value[2 * i * interval + j + interval] = get_value[1]
  return value


class BPDecoder:
  """BP 译码器"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self.info_indices = np.where(~self.frozen_bits)[0]
    self.G = build_generator_matrix(N)

  def decode(self, llr_ch):
    N = self.N
    n = self.n
    llr = np.asarray(llr_ch, dtype=np.float64)

    left_matrix = np.zeros((N, n + 1), dtype=np.float64)
    right_matrix = np.zeros((N, n + 1), dtype=np.float64)
    left_matrix[:, n] = llr
    right_matrix[:, 0] = np.where(self.frozen_bits, 1e6, 0.0)

    num_iters = 0
    u_hat = np.zeros(N, dtype=int)

    for it in range(1, self.max_iter + 1):
      for i in range(n):
        left_matrix[:, n - i - 1] = _bp_update_left(
          left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i, self.alpha
        )
      for i in range(n):
        right_matrix[:, i + 1] = _bp_update_right(
          left_matrix[:, i + 1], right_matrix[:, i], i + 1, self.alpha
        )

      u_llr = left_matrix[:, 0] + right_matrix[:, 0]
      for idx in range(N):
        if self.frozen_bits[idx]:
          u_hat[idx] = 0
        else:
          u_hat[idx] = 0 if u_llr[idx] >= 0 else 1

      x_hat = polar_encode(u_hat)
      x_llr = left_matrix[:, n] + right_matrix[:, n]
      x_hard = (x_llr < 0).astype(int)
      if np.array_equal(x_hat, x_hard):
        num_iters = it
        break
      num_iters = it

    u_llr = left_matrix[:, 0] + right_matrix[:, 0]
    for idx in range(N):
      if self.frozen_bits[idx]:
        u_hat[idx] = 0
      else:
        u_hat[idx] = 0 if u_llr[idx] >= 0 else 1

    return u_hat, num_iters
