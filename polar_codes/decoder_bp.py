"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
  """BP 译码器"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self._large = 1e6

  def _f_ms(self, a, b):
    return self.alpha * f_operation(a, b)

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    n = self.n
    N = self.N

    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)
    L[:, n] = llr_ch
    R[:, 0] = 0.0
    R[self.frozen_bits, 0] = self._large

    num_iters = 0
    u_hat = np.zeros(N, dtype=np.int32)

    for it in range(1, self.max_iter + 1):
      for j in range(n, 0, -1):
        step = 1 << (j - 1)
        for i in range(0, N, step * 2):
          L[i, j - 1] = self._f_ms(R[i, j] + L[i + step, j], L[i, j])
          L[i + step, j - 1] = self._f_ms(R[i, j], L[i, j]) + L[i + step, j]

      for j in range(0, n):
        step = 1 << j
        for i in range(0, N, step * 2):
          R[i, j + 1] = self._f_ms(
            R[i + step, j] + L[i + step, j + 1], R[i, j]
          )
          R[i + step, j + 1] = self._f_ms(R[i, j], L[i, j + 1]) + R[i + step, j]

      for i in range(N):
        u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
      u_hat[self.frozen_bits] = 0

      x_hat = polar_encode(u_hat)
      hard = (llr_ch < 0).astype(np.int32)
      if np.array_equal(x_hat, hard):
        num_iters = it
        break
      num_iters = it

    for i in range(N):
      u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
    u_hat[self.frozen_bits] = 0
    return u_hat, num_iters
