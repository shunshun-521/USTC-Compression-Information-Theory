"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from _ref_function import bp_update_left, bp_update_right, generate_matrix
from encoder import _bit_rev_indices, polar_encode


class BPDecoder:
  """BP 译码器。"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self.info_idx = np.where(~self.frozen_bits)[0]
    self.frozen_idx = np.where(self.frozen_bits)[0]
    self.G = generate_matrix(self.n)

  def decode(self, llr_ch):
    """主译码函数。"""
    N = self.N
    n = self.n
    brp = _bit_rev_indices(N)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)[brp]

    left_matrix = np.zeros((N, n + 1))
    right_matrix = np.zeros((N, n + 1))
    left_matrix[:, n] = llr_ch
    temp_value = (1 - 2 * self.frozen_bits.astype(int)) * np.inf
    temp = np.array([temp_value[i] if i in self.frozen_idx else 0 for i in range(N)])
    right_matrix[:, 0] = temp

    num_iters = 0
    u_hat = np.zeros(N, dtype=int)

    for it in range(self.max_iter):
      num_iters = it + 1
      for i in range(n):
        left_matrix[:, n - i - 1] = bp_update_left(
          left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i
        )
      for i in range(n):
        right_matrix[:, i + 1] = bp_update_right(
          left_matrix[:, i + 1], right_matrix[:, i], i + 1
        )

      u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
      u_hat = np.array([0 if u_d_llr[i] >= 0 else 1 for i in range(N)])
      u_hat[self.frozen_idx] = 0

      x_d_llr = left_matrix[:, n] + right_matrix[:, n]
      x_d = np.array([0 if x_d_llr[i] >= 0 else 1 for i in range(N)])
      x_g = (u_hat @ self.G) % 2
      if np.array_equal(x_g, x_d):
        break

    return u_hat, num_iters
