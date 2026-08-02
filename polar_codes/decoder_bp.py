"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
  """
  BP 译码器。
  因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
  """

  LARGE = 1e6

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self.frozen_idx = np.where(self.frozen_bits)[0]

  def _f_min_sum(self, a, b):
    return self.alpha * f_operation(a, b)

  def decode(self, llr_ch):
    """
    主译码函数。
    返回：(u_hat, num_iters)
    """
    N, n = self.N, self.n

    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)

    L[:, n] = llr_ch
    R[:, 0] = 0.0
    R[self.frozen_idx, 0] = self.LARGE

    num_iters = self.max_iter

    for it in range(1, self.max_iter + 1):
      for j in range(n, 0, -1):
        s = 1 << (j - 1)
        for i in range(0, N, 2 * s):
          for k in range(s):
            idx0 = i + k
            idx1 = i + k + s
            L[idx0, j - 1] = self._f_min_sum(
              R[idx0, j - 1] + L[idx1, j], L[idx0, j]
            )
            L[idx1, j - 1] = self._f_min_sum(
              R[idx0, j - 1], L[idx0, j]
            ) + L[idx1, j]

      for j in range(0, n):
        s = 1 << j
        for i in range(0, N, 2 * s):
          for k in range(s):
            idx0 = i + k
            idx1 = i + k + s
            R[idx0, j + 1] = self._f_min_sum(
              R[idx1, j + 1] + L[idx1, j + 1], R[idx0, j]
            )
            R[idx1, j + 1] = self._f_min_sum(
              R[idx0, j], L[idx0, j + 1]
            ) + R[idx1, j]

      u_hat = np.zeros(N, dtype=int)
      for i in range(N):
        total = L[i, 0] + R[i, 0]
        if self.frozen_bits[i]:
          u_hat[i] = 0
        else:
          u_hat[i] = 0 if total >= 0 else 1

      x_hat = polar_encode(u_hat)
      hard_ch = (llr_ch < 0).astype(int)
      if np.array_equal(x_hat, hard_ch):
        num_iters = it
        break
    else:
      u_hat = np.zeros(N, dtype=int)
      for i in range(N):
        total = L[i, 0] + R[i, 0]
        if self.frozen_bits[i]:
          u_hat[i] = 0
        else:
          u_hat[i] = 0 if total >= 0 else 1

    return u_hat, num_iters
