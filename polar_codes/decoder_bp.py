"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _f_ms(x, y, alpha):
  s1 = np.sign(x)
  s2 = np.sign(y)
  s1 = np.where(s1 == 0, 1, s1)
  s2 = np.where(s2 == 0, 1, s2)
  return alpha * s1 * s2 * np.minimum(np.abs(x), np.abs(y))


def _bp_update_left(left_col, right_col, layer, alpha):
  N = left_col.size
  interval = 2 ** (layer - 1)
  num = N // (interval * 2)
  value = np.zeros(N)
  for i in range(num):
    for j in range(interval):
      a = 2 * i * interval + j
      b = a + interval
      left_ele = np.array([left_col[a], left_col[b]])
      right_ele = np.array([right_col[a], right_col[b]])
      value[a] = _f_ms(right_ele[1] + left_ele[1], left_ele[0], alpha)
      value[b] = _f_ms(left_ele[0], right_ele[0], alpha) + left_ele[1]
  return value


def _bp_update_right(left_col, right_col, layer, alpha):
  N = left_col.size
  interval = 2 ** (layer - 1)
  num = N // (interval * 2)
  value = np.zeros(N)
  for i in range(num):
    for j in range(interval):
      a = 2 * i * interval + j
      b = a + interval
      left_ele = np.array([left_col[a], left_col[b]])
      right_ele = np.array([right_col[a], right_col[b]])
      value[a] = _f_ms(right_ele[1] + left_ele[1], right_ele[0], alpha)
      value[b] = _f_ms(left_ele[0], right_ele[0], alpha) + right_ele[1]
  return value


class BPDecoder:
  """BP 译码器"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self.large = 1e6

  def decode(self, llr_ch):
    """主译码函数"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = self.N
    n = self.n

    L = np.zeros((N, n + 1))
    R = np.zeros((N, n + 1))
    L[:, n] = llr_ch
    R[:, 0] = 0.0
    R[self.frozen_bits, 0] = self.large

    num_iters = 0
    u_hat = np.zeros(N, dtype=int)

    for it in range(1, self.max_iter + 1):
      for i in range(n):
        L[:, n - i - 1] = _bp_update_left(L[:, n - i], R[:, n - i - 1], n - i, self.alpha)
      for i in range(n):
        R[:, i + 1] = _bp_update_right(L[:, i + 1], R[:, i], i + 1, self.alpha)

      num_iters = it
      total_llr = L[:, 0] + R[:, 0]
      u_hat = (total_llr < 0).astype(int)
      u_hat[self.frozen_bits] = 0

      x_hat = polar_encode(u_hat)
      hard_x = (llr_ch < 0).astype(int)
      if np.array_equal(x_hat, hard_x):
        break

    total_llr = L[:, 0] + R[:, 0]
    u_hat = (total_llr < 0).astype(int)
    u_hat[self.frozen_bits] = 0
    return u_hat, num_iters
