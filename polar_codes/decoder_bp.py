"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_minsum(x, y, alpha):
  return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
  """BP 译码器。"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self.large = 1e10

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = self.N
    n = self.n

    # L[stage, i]: 从右向左消息；R[stage, i]: 从左向右消息
    L = np.zeros((n + 1, N), dtype=np.float64)
    R = np.zeros((n + 1, N), dtype=np.float64)
    L[n, :] = llr_ch
    R[0, :] = 0.0
    R[0, self.frozen_bits] = self.large

    num_iters = 0
    for _ in range(self.max_iter):
      num_iters += 1

      for stage in range(n):
        step = 1 << stage
        for block in range(0, N, 2 * step):
          for j in range(step):
            i1 = block + j
            i2 = block + j + step
            R[stage + 1, i1] = _f_minsum(
              R[stage, i2] + L[stage + 1, i2],
              R[stage, i1],
              self.alpha,
            )
            R[stage + 1, i2] = (
              _f_minsum(R[stage, i1], L[stage + 1, i1], self.alpha)
              + R[stage, i2]
            )

      for stage in range(n - 1, -1, -1):
        step = 1 << stage
        for block in range(0, N, 2 * step):
          for j in range(step):
            i1 = block + j
            i2 = block + j + step
            L[stage, i1] = _f_minsum(
              R[stage, i1] + L[stage + 1, i2],
              L[stage + 1, i1],
              self.alpha,
            )
            L[stage, i2] = (
              _f_minsum(R[stage, i1], L[stage + 1, i1], self.alpha)
              + L[stage + 1, i2]
            )

      u_hat = np.zeros(N, dtype=int)
      for i in range(N):
        if self.frozen_bits[i]:
          u_hat[i] = 0
        else:
          total = L[0, i] + R[0, i]
          u_hat[i] = 0 if total >= 0 else 1

      x_hat = polar_encode(u_hat)
      hard_ch = (llr_ch < 0).astype(int)
      if np.array_equal(x_hat, hard_ch):
        break

    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
      if self.frozen_bits[i]:
        u_hat[i] = 0
      else:
        total = L[0, i] + R[0, i]
        u_hat[i] = 0 if total >= 0 else 1

    return u_hat, num_iters
