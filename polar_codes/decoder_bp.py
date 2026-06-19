"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _channel_llr_for_decode
from encoder import polar_encode


def _element_update_left(left, right, alpha=0.9375):
  value = np.zeros(2)
  value[0] = alpha * np.sign(right[1] + left[1]) * np.sign(left[0]) * min(
      abs(right[1] + left[1]), abs(left[0])
  )
  value[1] = alpha * np.sign(left[0]) * np.sign(right[0]) * min(
      abs(left[0]), abs(right[0])
  ) + left[1]
  return value


def _element_update_right(left, right, alpha=0.9375):
  value = np.zeros(2)
  value[0] = alpha * np.sign(right[1] + left[1]) * np.sign(right[0]) * min(
      abs(right[1] + left[1]), abs(right[0])
  )
  value[1] = alpha * np.sign(left[0]) * np.sign(right[0]) * min(
      abs(left[0]), abs(right[0])
  ) + right[1]
  return value


def _bp_update_left(left_array, right_array, layer_n, alpha=0.9375):
  N = left_array.size
  interval = 2 ** (layer_n - 1)
  num = N // (interval * 2)
  value = np.zeros(N)
  for i in range(num):
    for j in range(interval):
      idx = 2 * i * interval + j
      left_ele = np.array([left_array[idx], left_array[idx + interval]])
      right_ele = np.array([right_array[idx], right_array[idx + interval]])
      out = _element_update_left(left_ele, right_ele, alpha)
      value[idx] = out[0]
      value[idx + interval] = out[1]
  return value


def _bp_update_right(left_array, right_array, layer_n, alpha=0.9375):
  N = left_array.size
  interval = 2 ** (layer_n - 1)
  num = N // (interval * 2)
  value = np.zeros(N)
  for i in range(num):
    for j in range(interval):
      idx = 2 * i * interval + j
      left_ele = np.array([left_array[idx], left_array[idx + interval]])
      right_ele = np.array([right_array[idx], right_array[idx + interval]])
      out = _element_update_right(left_ele, right_ele, alpha)
      value[idx] = out[0]
      value[idx + interval] = out[1]
  return value


class BPDecoder:
  """BP 译码器。"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self._large = 1e6

  def decode(self, llr_ch):
    llr_ch = _channel_llr_for_decode(llr_ch)
    N, n = self.N, self.n

    left_matrix = np.zeros((N, n + 1))
    right_matrix = np.zeros((N, n + 1))
    left_matrix[:, n] = llr_ch
    right_matrix[:, 0] = np.where(
        self.frozen_bits, self._large, 0.0
    )

    for num_iters in range(1, self.max_iter + 1):
      for i in range(n):
        left_matrix[:, n - i - 1] = _bp_update_left(
            left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i, self.alpha
        )

      for i in range(n):
        right_matrix[:, i + 1] = _bp_update_right(
            left_matrix[:, i + 1], right_matrix[:, i], i + 1, self.alpha
        )

      u_llr = left_matrix[:, 0] + right_matrix[:, 0]
      u_hat = np.zeros(N, dtype=int)
      for i in range(N):
        if self.frozen_bits[i]:
          u_hat[i] = 0
        else:
          u_hat[i] = 0 if u_llr[i] >= 0 else 1

      x_hat = polar_encode(u_hat)
      hard_ch = (llr_ch < 0).astype(int)
      if np.array_equal(x_hat, hard_ch):
        return u_hat, num_iters

    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
      if self.frozen_bits[i]:
        u_hat[i] = 0
      else:
        u_hat[i] = 0 if u_llr[i] >= 0 else 1
    return u_hat, self.max_iter
