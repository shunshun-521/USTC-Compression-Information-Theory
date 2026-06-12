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
    self.large = 1e6

  def decode(self, llr_ch):
    N, n = self.N, self.n
    alpha = self.alpha

    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)
    L[:, n] = llr_ch
    R[:, 0] = 0.0
    R[self.frozen_bits, 0] = self.large

    num_iters = self.max_iter

    for it in range(1, self.max_iter + 1):
      # 右到左更新 L
      for j in range(n, 0, -1):
        s = 1 << (j - 1)
        for i in range(0, N, 2 * s):
          for k in range(s):
            li, ri = i + k, i + k + s
            L[li, j - 1] = _f_minsum(R[li, j] + L[ri, j], L[li, j], alpha)
            L[ri, j - 1] = _f_minsum(R[li, j], L[li, j], alpha) + L[ri, j]

      # 左到右更新 R
      for j in range(0, n):
        s = 1 << j
        for i in range(0, N, 2 * s):
          for k in range(s):
            li, ri = i + k, i + k + s
            R[li, j + 1] = _f_minsum(R[ri, j] + L[ri, j + 1], R[li, j], alpha)
            R[ri, j + 1] = _f_minsum(R[li, j], L[li, j + 1], alpha) + R[ri, j]

      # 早停
      total = L[:, 0] + R[:, 0]
      u_hat = np.where(total >= 0, 0, 1).astype(int)
      u_hat[self.frozen_bits] = 0
      x_hat = polar_encode(u_hat)
      hard_ch = (llr_ch < 0).astype(int)
      if np.array_equal(x_hat, hard_ch):
        num_iters = it
        break

    total = L[:, 0] + R[:, 0]
    u_hat = np.where(total >= 0, 0, 1).astype(int)
    u_hat[self.frozen_bits] = 0
    return u_hat, num_iters
