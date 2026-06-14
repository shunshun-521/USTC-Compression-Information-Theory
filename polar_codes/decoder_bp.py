"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


def _ms(a, b, alpha=0.9375):
  return alpha * f_operation(a, b)


def _bp_update_left(left_col, right_col, layer_n, alpha):
  N = left_col.size
  interval = 2 ** (layer_n - 1)
  num = N // (interval * 2)
  value = np.zeros(N, dtype=np.float64)
  for i in range(num):
    for j in range(interval):
      base = 2 * i * interval + j
      left0, left1 = left_col[base], left_col[base + interval]
      right0, right1 = right_col[base], right_col[base + interval]
      value[base] = _ms(right1 + left1, left0, alpha)
      value[base + interval] = _ms(left0, right0, alpha) + left1
  return value


def _bp_update_right(left_col, right_col, layer_n, alpha):
  N = left_col.size
  interval = 2 ** (layer_n - 1)
  num = N // (interval * 2)
  value = np.zeros(N, dtype=np.float64)
  for i in range(num):
    for j in range(interval):
      base = 2 * i * interval + j
      left0, left1 = left_col[base], left_col[base + interval]
      right0, right1 = right_col[base], right_col[base + interval]
      value[base] = _ms(right1 + left1, right0, alpha)
      value[base + interval] = _ms(left0, right0, alpha) + right1
  return value


class BPDecoder:
  """BP 译码器"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=np.int32)
    self.max_iter = max_iter
    self.alpha = alpha
    self._large = 1e6

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    n = self.n
    N = self.N

    left = np.zeros((N, n + 1), dtype=np.float64)
    right = np.zeros((N, n + 1), dtype=np.float64)
    left[:, n] = llr_ch

    for i in range(N):
      right[i, 0] = self._large if self.frozen_bits[i] else 0.0

    num_iters = 0
    u_hat = np.zeros(N, dtype=np.int32)

    for it in range(1, self.max_iter + 1):
      for i in range(n):
        left[:, n - i - 1] = _bp_update_left(
          left[:, n - i], right[:, n - i - 1], n - i, self.alpha
        )
      for i in range(n):
        right[:, i + 1] = _bp_update_right(
          left[:, i + 1], right[:, i], i + 1, self.alpha
        )

      posterior = left[:, 0] + right[:, 0]
      u_hat = np.where(posterior >= 0, 0, 1).astype(np.int32)
      u_hat[self.frozen_bits.astype(bool)] = 0

      x_hat = polar_encode(u_hat)
      hard = (llr_ch < 0).astype(np.int32)
      num_iters = it
      if np.array_equal(x_hat, hard):
        break

    posterior = left[:, 0] + right[:, 0]
    u_hat = np.where(posterior >= 0, 0, 1).astype(np.int32)
    u_hat[self.frozen_bits.astype(bool)] = 0
    return u_hat, num_iters
