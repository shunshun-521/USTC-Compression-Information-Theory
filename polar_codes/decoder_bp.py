"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
  """BP 译码器"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha

  def _f_ms(self, x, y):
    return self.alpha * f_operation(x, y)

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N, n = self.N, self.n
    large = 1e6

    L = np.zeros((n + 1, N), dtype=np.float64)
    R = np.zeros((n + 1, N), dtype=np.float64)
    L[n, :] = llr_ch
    R[0, self.frozen_bits] = large

    num_iters = self.max_iter
    u_hat = np.zeros(N, dtype=int)

    for it in range(1, self.max_iter + 1):
      for col in range(n - 1, -1, -1):
        step = 2 ** col
        for block in range(0, N, 2 * step):
          for k in range(step):
            i = block + k
            j = i + step
            L[col, i] = self._f_ms(R[col + 1, i] + L[col + 1, j], L[col + 1, i])
            L[col, j] = self._f_ms(R[col + 1, i], L[col + 1, i]) + L[col + 1, j]

      for col in range(1, n + 1):
        step = 2 ** (col - 1)
        for block in range(0, N, 2 * step):
          for k in range(step):
            i = block + k
            j = i + step
            R[col, i] = self._f_ms(R[col, j] + L[col, j], R[col - 1, i])
            R[col, j] = self._f_ms(R[col - 1, i], L[col, i]) + R[col, j]

      for idx in range(N):
        total = L[0, idx] + R[0, idx]
        u_hat[idx] = 0 if self.frozen_bits[idx] or total >= 0 else 1

      x_hat = polar_encode(u_hat)
      hard_ch = (llr_ch < 0).astype(int)
      if np.array_equal(x_hat, hard_ch):
        num_iters = it
        break

    for idx in range(N):
      total = L[0, idx] + R[0, idx]
      u_hat[idx] = 0 if self.frozen_bits[idx] or total >= 0 else 1

    return u_hat, num_iters
