"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha=0.9375):
  s1 = 1 if a == 0 else np.sign(a)
  s2 = 1 if b == 0 else np.sign(b)
  return alpha * s1 * s2 * min(abs(a), abs(b))


def _bp_update_left(left_col, right_col, stage):
  """左向 LLR 更新（stage 为 1..n）。"""
  N = len(left_col)
  interval = 2 ** (stage - 1)
  num = N // (interval * 2)
  value = np.zeros(N, dtype=np.float64)
  for i in range(num):
    for j in range(interval):
      idx0 = 2 * i * interval + j
      idx1 = idx0 + interval
      l0, l1 = left_col[idx0], left_col[idx1]
      r0, r1 = right_col[idx0], right_col[idx1]
      value[idx0] = _f_min_sum(r1 + l1, l0)
      value[idx1] = _f_min_sum(l0, r0) + l1
  return value


def _bp_update_right(left_col, right_col, stage):
  """右向 LLR 更新（stage 为 1..n）。"""
  N = len(left_col)
  interval = 2 ** (stage - 1)
  num = N // (interval * 2)
  value = np.zeros(N, dtype=np.float64)
  for i in range(num):
    for j in range(interval):
      idx0 = 2 * i * interval + j
      idx1 = idx0 + interval
      l0, l1 = left_col[idx0], left_col[idx1]
      r0, r1 = right_col[idx0], right_col[idx1]
      value[idx0] = _f_min_sum(r1 + l1, r0)
      value[idx1] = _f_min_sum(l0, r0) + r1
  return value


class BPDecoder:
  """BP 译码器。"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self.rev = bit_reversal_permutation(N)
    self.LARGE = 1e8

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.rev]
    N = self.N
    n = self.n

    left = np.zeros((N, n + 1), dtype=np.float64)
    right = np.zeros((N, n + 1), dtype=np.float64)
    left[:, n] = llr_ch

    for i in range(N):
      if self.frozen_bits[i]:
        right[i, 0] = self.LARGE
      else:
        right[i, 0] = 0.0

    num_iters = 0
    for it in range(self.max_iter):
      num_iters = it + 1
      for stage in range(n):
        left[:, n - stage - 1] = _bp_update_left(
          left[:, n - stage],
          right[:, n - stage - 1],
          n - stage,
        )
      for stage in range(n):
        right[:, stage + 1] = _bp_update_right(
          left[:, stage + 1],
          right[:, stage],
          stage + 1,
        )

      u_hat = self._decide(left, right)
      x_hat = polar_encode(u_hat)
      hard_ch = (llr_ch < 0).astype(int)
      if np.array_equal(x_hat, hard_ch):
        break

    u_hat = self._decide(left, right)
    return u_hat, num_iters

  def _decide(self, left, right):
    total = left[:, 0] + right[:, 0]
    u_hat = np.zeros(self.N, dtype=int)
    for i in range(self.N):
      if self.frozen_bits[i]:
        u_hat[i] = 0
      else:
        u_hat[i] = 0 if total[i] >= 0 else 1
    return u_hat
