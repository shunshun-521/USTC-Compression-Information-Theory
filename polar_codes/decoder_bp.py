"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode

LARGE = 1e6


def _min_sum_f(a, b, alpha):
  return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
  """BP 译码器。"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self.rev = bit_reversal_permutation(N)

  def _f_min_sum(self, a, b):
    return _min_sum_f(a, b, self.alpha)

  def decode(self, llr_ch):
    """主译码函数，返回 (u_hat, num_iters)。"""
    N, n = self.N, self.n
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    rev_llr = llr_ch[self.rev]

    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)
    L[:, n] = rev_llr
    R[:, 0] = 0.0
    R[self.frozen_bits, 0] = LARGE

    num_iters = self.max_iter

    for it in range(1, self.max_iter + 1):
      for j in range(n, 0, -1):
        s = 1 << (j - 1)
        for i in range(0, N, 2 * s):
          L[i, j - 1] = self._f_min_sum(R[i, j] + L[i + s, j], L[i, j])
          L[i + s, j - 1] = self._f_min_sum(R[i, j], L[i, j]) + L[i + s, j]

      for j in range(0, n):
        s = 1 << j
        for i in range(0, N, 2 * s):
          R[i, j + 1] = self._f_min_sum(R[i + s, j] + L[i + s, j + 1], R[i, j])
          R[i + s, j + 1] = self._f_min_sum(R[i, j], L[i, j + 1]) + R[i + s, j]

      total = L[:, 0] + R[:, 0]
      u_hat = (total < 0).astype(int)
      u_hat[self.frozen_bits] = 0

      x_hat = polar_encode(u_hat)
      hard_ch = (llr_ch < 0).astype(int)
      if np.array_equal(x_hat, hard_ch):
        num_iters = it
        break

    total = L[:, 0] + R[:, 0]
    u_hat = (total < 0).astype(int)
    u_hat[self.frozen_bits] = 0
    return u_hat.astype(int), num_iters
