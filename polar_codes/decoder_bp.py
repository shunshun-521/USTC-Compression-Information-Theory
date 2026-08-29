"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from _ref.decoder_utils import lower_llr, upper_llr
from encoder import polar_encode


class BPDecoder:
  """BP 译码器"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self.frozen_idx = np.where(self.frozen_bits)[0]
    self.LARGE = 1e6

  def _f(self, x, y):
    sign = np.sign(x) * np.sign(y)
    mag = np.minimum(np.abs(x), np.abs(y))
    return self.alpha * sign * mag

  def decode(self, llr_ch):
    n = self.n
    N = self.N

  # L[i,j]: right-to-left, R[i,j]: left-to-right
    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)
    L[:, n] = llr_ch.astype(np.float64)
    R[:, 0] = 0.0
    R[self.frozen_idx, 0] = self.LARGE

    num_iters = self.max_iter
    u_hat = np.zeros(N, dtype=int)

    for it in range(1, self.max_iter + 1):
      for stage in range(n):
        stride = 1 << stage
        for base in range(0, N, 2 * stride):
          for offset in range(stride):
            i = base + offset
            j = n - stage - 1
            jp = j + 1
            L[i, j] = self._f(L[i, jp], R[i + stride, j] + L[i + stride, jp])
            L[i + stride, j] = self._f(L[i, jp], R[i, j]) + L[i + stride, jp]

      for stage in range(n):
        stride = 1 << stage
        for base in range(0, N, 2 * stride):
          for offset in range(stride):
            i = base + offset
            j = stage
            R[i, j + 1] = self._f(R[i, j], R[i + stride, j] + L[i + stride, j + 1])
            R[i + stride, j + 1] = self._f(R[i, j], L[i, j + 1]) + R[i + stride, j]

      for i in range(N):
        total = L[i, 0] + R[i, 0]
        u_hat[i] = 0 if (total >= 0 or self.frozen_bits[i]) else 1

      x_hat = polar_encode(u_hat)
      hard_ch = (llr_ch < 0).astype(int)
      if np.array_equal(x_hat, hard_ch):
        num_iters = it
        break

    for i in range(N):
      total = L[i, 0] + R[i, 0]
      u_hat[i] = 0 if (total >= 0 or self.frozen_bits[i]) else 1

    return u_hat, num_iters
