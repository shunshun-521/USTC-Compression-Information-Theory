"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def bp_f_ms(x, y, alpha=0.9375):
  """min-sum 近似的 f 运算"""
  return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
  """BP 译码器"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self.frozen_idx = np.where(self.frozen_bits)[0]
    self.info_idx = np.where(~self.frozen_bits)[0]
    self.large = 1e6
    self.bit_rev = bit_reversal_permutation(N)

  def _hard_decision(self, L, R):
    total = L[0, :] + R[0, :]
    u_hat = np.zeros(self.N, dtype=np.int8)
    u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(np.int8)
    return u_hat

  def decode(self, llr_ch):
    n = self.n
    N = self.N
    llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.bit_rev]

    L = np.zeros((n + 1, N), dtype=np.float64)
    R = np.zeros((n + 1, N), dtype=np.float64)
    L[n, :] = llr_ch
    R[0, :] = 0.0
    R[0, self.frozen_idx] = self.large

    num_iters = 0
    for it in range(1, self.max_iter + 1):
      num_iters = it

      for j in range(n, 0, -1):
        s = 1 << (j - 1)
        for i in range(0, N, 2 * s):
          for k in range(s):
            idx = i + k
            L[j - 1, idx] = bp_f_ms(
              R[j, idx] + L[j, idx + s], L[j, idx], self.alpha
            )
            L[j - 1, idx + s] = bp_f_ms(R[j, idx], L[j, idx], self.alpha) + L[
              j, idx + s
            ]

      for j in range(0, n):
        s = 1 << j
        for i in range(0, N, 2 * s):
          for k in range(s):
            idx = i + k
            R[j + 1, idx] = bp_f_ms(
              R[j, idx + s] + L[j + 1, idx + s], R[j, idx], self.alpha
            )
            R[j + 1, idx + s] = bp_f_ms(R[j, idx], L[j + 1, idx], self.alpha) + R[
              j, idx + s
            ]

      u_hat = self._hard_decision(L, R)
      x_hat = polar_encode(u_hat)
      x_hard = (llr_ch < 0).astype(np.int8)
      if np.array_equal(x_hat, x_hard):
        break

    u_hat = self._hard_decision(L, R)
    return u_hat, num_iters
