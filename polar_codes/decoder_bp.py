"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _boxplus(x, y, llr_max=30.0):
  """精确 boxplus 运算"""
  x = np.clip(x, -llr_max, llr_max)
  y = np.clip(y, -llr_max, llr_max)
  return np.log1p(np.exp(x + y)) - np.logaddexp(x, y)


def _f_min_sum(a, b, alpha=0.9375):
  """min-sum f 运算，带 alpha 修正"""
  return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
  """BP 译码器"""

  LARGE = 1e6

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, use_min_sum=True):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.max_iter = max_iter
    self.alpha = alpha
    self.use_min_sum = use_min_sum
    self.frozen_idx = np.where(self.frozen_bits)[0]
    self.rev = bit_reversal_permutation(N)
    self._reset_state()

  def _reset_state(self):
    N, n = self.N, self.n
    self.msg_l = [None] * (n + 1)
    self.msg_r = [None] * (n + 1)
    self.msg_r_in = np.zeros(N, dtype=np.float64)
    self.msg_r_in[self.frozen_idx] = self.LARGE
    self._iteration = 0

  def _cn_op(self, x, y):
    if self.use_min_sum:
      return _f_min_sum(x, y, self.alpha)
    return _boxplus(x, y)

  def _bp_iteration(self, llr_ch):
    """执行一次完整的左右 BP 迭代"""
    N, n = self.N, self.n
    ind_it = self._iteration

    for ind_s in range(n):
      ind_range = np.arange(N // 2)
      ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
      ind_2 = ind_1 + 2 ** ind_s
      ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

      if ind_s == n - 1:
        l1_in = llr_ch[ind_1]
        l2_in = llr_ch[ind_2]
      elif ind_it == 0:
        l1_in = np.zeros(N // 2)
        l2_in = np.zeros(N // 2)
      else:
        l_in = self.msg_l[ind_s + 1]
        l1_in = l_in[ind_1]
        l2_in = l_in[ind_2]

      if ind_s == 0:
        r1_in = self.msg_r_in[ind_1]
        r2_in = self.msg_r_in[ind_2]
      else:
        r_in = self.msg_r[ind_s]
        r1_in = r_in[ind_1]
        r2_in = r_in[ind_2]

      r1_out = self._cn_op(r1_in, l2_in + r2_in)
      r2_out = self._cn_op(r1_in, l1_in) + r2_in
      self.msg_r[ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

    for ind_s in range(n - 1, -1, -1):
      ind_range = np.arange(N // 2)
      ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
      ind_2 = ind_1 + 2 ** ind_s
      ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

      if ind_s == n - 1:
        l1_in = llr_ch[ind_1]
        l2_in = llr_ch[ind_2]
      else:
        l_in = self.msg_l[ind_s + 1]
        l1_in = l_in[ind_1]
        l2_in = l_in[ind_2]

      if ind_s == 0:
        r1_in = self.msg_r_in[ind_1]
        r2_in = self.msg_r_in[ind_2]
      else:
        r_in = self.msg_r[ind_s]
        r1_in = r_in[ind_1]
        r2_in = r_in[ind_2]

      l1_out = self._cn_op(l1_in, l2_in + r2_in)
      l2_out = self._cn_op(r1_in, l1_in) + l2_in
      self.msg_l[ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

    self._iteration += 1
    return self.msg_l[0]

  def decode(self, llr_ch):
    """
    主译码函数。
    返回：(u_hat, num_iters)
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    llr_ch = llr_ch[self.rev]
    self._reset_state()
    num_iters = 0
    u_hat = np.zeros(self.N, dtype=int)

    for it in range(1, self.max_iter + 1):
      l_stage0 = self._bp_iteration(llr_ch)
      for i in range(self.N):
        if self.frozen_bits[i]:
          u_hat[i] = 0
        else:
          u_hat[i] = 0 if l_stage0[i] >= 0 else 1

      x_hat = polar_encode(u_hat)
      x_hard = (llr_ch < 0).astype(int)
      num_iters = it
      if np.array_equal(x_hat, x_hard):
        break

    return u_hat, num_iters
