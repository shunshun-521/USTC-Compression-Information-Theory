"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from channel import hard_decision_llr
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
    self.LARGE = 1e6

  def _f_min_sum(self, a, b):
    return self.alpha * f_operation(a, b)

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    n, N = self.n, self.N

    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)
    L[:, n] = llr_ch
    R[:, :] = 0.0
    R[self.frozen_bits, 0] = self.LARGE

    num_iters = 0
    u_hat = np.zeros(N, dtype=int)

    for it in range(1, self.max_iter + 1):
      num_iters = it
      L_new = L.copy()
      R_new = R.copy()

      for j in range(n, 0, -1):
        s = 1 << (j - 1)
        for i in range(0, N, 2 * s):
          L_new[i, j - 1] = self._f_min_sum(
              R[i, j] + L[i + s, j], L[i, j]
          )
          L_new[i + s, j - 1] = self._f_min_sum(
              R[i, j], L[i, j]
          ) + L[i + s, j]

      for j in range(0, n):
        s = 1 << j
        for i in range(0, N, 2 * s):
          R_new[i, j + 1] = self._f_min_sum(
              R[i + s, j] + L[i + s, j + 1], R[i, j]
          )
          R_new[i + s, j + 1] = self._f_min_sum(
              R[i, j], L[i, j + 1]
          ) + R[i + s, j]

      L[:, :n] = L_new[:, :n]
      R[:, 1:] = R_new[:, 1:]
      L[:, n] = llr_ch
      R[self.frozen_bits, 0] = self.LARGE

      for i in range(N):
        if self.frozen_bits[i]:
          u_hat[i] = 0
        else:
          u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

      x_hat = polar_encode(u_hat)
      x_hard = hard_decision_llr(llr_ch)
      if np.array_equal(x_hat, x_hard):
        break

    for i in range(N):
      if self.frozen_bits[i]:
        u_hat[i] = 0
      else:
        u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

    return u_hat, num_iters
