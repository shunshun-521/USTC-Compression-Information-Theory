"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _minsum_f(a, b, alpha=0.9375):
  """min-sum f 运算"""
  return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
  """BP 译码器"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self.large = 1e6
    self.rev = bit_reversal_permutation(N)

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    n = self.n
    N = self.N
    llr_u = llr_ch[self.rev]

    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)

    L[:, n] = llr_u
    R[:, 0] = 0.0
    R[self.frozen_bits, 0] = self.large

    num_iters = 0
    u_hat = np.zeros(N, dtype=int)

    for it in range(self.max_iter):
      num_iters = it + 1

      for j in range(n, 0, -1):
        s = 1 << (j - 1)
        for i in range(0, N, 2 * s):
          for k in range(s):
            li = i + k
            li_s = i + k + s
            L[li, j - 1] = _minsum_f(
              R[li, j] + L[li_s, j + 1], L[li, j + 1], self.alpha,
            )
            L[li_s, j - 1] = (
              _minsum_f(R[li, j], L[li, j + 1], self.alpha) + L[li_s, j + 1]
            )

      for j in range(0, n):
        s = 1 << j
        for i in range(0, N, 2 * s):
          for k in range(s):
            li = i + k
            li_s = i + k + s
            R[li, j + 1] = _minsum_f(
              R[li_s, j + 1] + L[li_s, j + 1], R[li, j], self.alpha,
            )
            R[li_s, j + 1] = (
              _minsum_f(R[li, j], L[li, j + 1], self.alpha) + R[li_s, j + 1]
            )

      for i in range(N):
        total = L[i, 0] + R[i, 0]
        u_hat[i] = 0 if total >= 0 else 1
      u_hat[self.frozen_bits] = 0

      x_hat = polar_encode(u_hat)
      hard_ch = (llr_ch < 0).astype(int)
      if np.array_equal(x_hat, hard_ch):
        break

    for i in range(N):
      total = L[i, 0] + R[i, 0]
      u_hat[i] = 0 if total >= 0 else 1
    u_hat[self.frozen_bits] = 0

    return u_hat, num_iters
