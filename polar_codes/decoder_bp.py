"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _boxplus(a, b):
  """LLR 域 box-plus（f 运算的精确形式）。"""
  if a > 1e20 or b > 1e20:
    return min(a, b)
  if a < -1e20 or b < -1e20:
    return -min(abs(a), abs(b))
  ta = np.tanh(a / 2.0)
  tb = np.tanh(b / 2.0)
  prod = np.clip(ta * tb, -1.0 + 1e-12, 1.0 - 1e-12)
  return 2.0 * np.arctanh(prod)


def _minsum_f(a, b, alpha=0.9375):
  sa = 1.0 if a >= 0 else -1.0
  sb = 1.0 if b >= 0 else -1.0
  return alpha * sa * sb * min(abs(a), abs(b))


class BPDecoder:
  """BP 译码器。"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self.frozen_idx = np.where(self.frozen_bits)[0]
    self.br = bit_reversal_permutation(N)
    self._large = 1e6

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    llr_perm = llr_ch[self.br]
    n = self.n
    N = self.N

    # 分层消息：layer 0..n，每层 N 个节点
    L = np.zeros((n + 1, N), dtype=np.float64)
    R = np.zeros((n + 1, N), dtype=np.float64)
    L[n, :] = llr_perm
    R[0, self.frozen_idx] = self._large

    num_iters = self.max_iter

    for it in range(1, self.max_iter + 1):
      # 右到左
      for layer in range(n - 1, -1, -1):
        step = 1 << layer
        for block in range(0, N, 2 * step):
          for i in range(step):
            a = block + i
            b = block + i + step
            L[layer, a] = _boxplus(
              R[layer + 1, a] + L[layer + 1, b], L[layer + 1, a]
            )
            L[layer, b] = _boxplus(
              R[layer + 1, a], L[layer + 1, a]
            ) + L[layer + 1, b]

      # 左到右
      for layer in range(0, n):
        step = 1 << layer
        for block in range(0, N, 2 * step):
          for i in range(step):
            a = block + i
            b = block + i + step
            R[layer + 1, a] = _boxplus(
              R[layer, b] + L[layer + 1, b], R[layer, a]
            )
            R[layer + 1, b] = _boxplus(
              R[layer, a], L[layer + 1, a]
            ) + R[layer, b]

      u_hat = self._hard_decision(L, R)
      x_hat = polar_encode(u_hat)
      if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
        num_iters = it
        break

    u_hat = self._hard_decision(L, R)
    return u_hat, num_iters

  def _hard_decision(self, L, R):
    total = L[0, :] + R[0, :]
    u_hat = (total < 0).astype(int)
    u_hat[self.frozen_idx] = 0
    return u_hat
