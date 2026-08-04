"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation, channel_llr_for_decode
from encoder import polar_encode


class BPDecoder:
  """BP 译码器"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self.LARGE = 1e6

  def _f_min_sum(self, a, b):
    return self.alpha * f_operation(a, b)

  def decode(self, llr_ch):
    """主译码函数"""
    N = self.N
    n = self.n
    llr_ch = np.asarray(llr_ch, dtype=np.float64)

    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)

    L[:, n] = channel_llr_for_decode(llr_ch)
    R[:, 0] = 0.0
    R[self.frozen_bits, 0] = self.LARGE

    num_iters = 0
    for it in range(self.max_iter):
      num_iters = it + 1

      for j in range(n, 0, -1):
        s = 1 << (j - 1)
        for i in range(0, N, 2 * s):
          for k in range(s):
            idx1 = i + k
            idx2 = i + s + k
            L[idx1, j - 1] = self._f_min_sum(
              R[idx1, j] + L[idx2, j], L[idx1, j]
            )
            L[idx2, j - 1] = self._f_min_sum(R[idx1, j], L[idx1, j]) + L[idx2, j]

      for j in range(0, n):
        s = 1 << j
        for i in range(0, N, 2 * s):
          for k in range(s):
            idx1 = i + k
            idx2 = i + s + k
            R[idx1, j + 1] = self._f_min_sum(
              R[idx2, j] + L[idx2, j + 1], R[idx1, j]
            )
            R[idx2, j + 1] = self._f_min_sum(R[idx1, j], L[idx1, j + 1]) + R[idx2, j]

      total_llr = L[:, 0] + R[:, 0]
      u_hat = np.zeros(N, dtype=int)
      u_hat[~self.frozen_bits] = (total_llr[~self.frozen_bits] < 0).astype(int)
      u_hat[self.frozen_bits] = 0

      x_hat = polar_encode(u_hat)
      hard_ch = (llr_ch < 0).astype(int)
      if np.array_equal(x_hat, hard_ch):
        break

    total_llr = L[:, 0] + R[:, 0]
    u_hat = np.zeros(N, dtype=int)
    u_hat[~self.frozen_bits] = (total_llr[~self.frozen_bits] < 0).astype(int)
    u_hat[self.frozen_bits] = 0

    return u_hat, num_iters
