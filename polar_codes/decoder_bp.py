"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
  """BP 译码器（参考 Sionna/Arikan 因子图消息传递结构）"""

  def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.frozen_pos = np.where(self.frozen_bits)[0]
    self.max_iter = max_iter
    self.alpha = alpha
    self.llr_max = 19.3
    self.br = bit_reversal_permutation(N)

  def _f_min_sum(self, a, b):
    a = np.clip(a, -self.llr_max, self.llr_max)
    b = np.clip(b, -self.llr_max, self.llr_max)
    return self.alpha * f_operation(a, b)

  def _stage_indices(self, stage):
    ind_range = np.arange(self.N // 2)
    ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
    ind_2 = ind_1 + 2 ** stage
    ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
    return ind_1, ind_2, ind_inv

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    llr_work = llr_ch[self.br]
    n = self.n
    N = self.N

    msg_l = [[None] * (n + 1) for _ in range(self.max_iter)]
    msg_r = [[None] * (n + 1) for _ in range(self.max_iter)]

    msg_r_in = np.zeros(N, dtype=np.float64)
    msg_r_in[self.frozen_pos] = self.llr_max

    num_iters = 0
    for it in range(self.max_iter):
      num_iters = it + 1

      # 左到右更新 R
      for s in range(n):
        ind_1, ind_2, ind_inv = self._stage_indices(s)
        if s == n - 1:
          l1_in = llr_work[ind_1]
          l2_in = llr_work[ind_2]
        elif it == 0:
          l1_in = np.zeros(N // 2)
          l2_in = np.zeros(N // 2)
        else:
          l_in = msg_l[it - 1][s + 1]
          l1_in = l_in[ind_1]
          l2_in = l_in[ind_2]

        if s == 0:
          r1_in = msg_r_in[ind_1]
          r2_in = msg_r_in[ind_2]
        else:
          r_in = msg_r[it][s]
          r1_in = r_in[ind_1]
          r2_in = r_in[ind_2]

        r1_out = self._f_min_sum(r1_in, l2_in + r2_in)
        r2_out = self._f_min_sum(r1_in, l1_in) + r2_in
        r_out = np.empty(N)
        r_out[ind_1] = r1_out
        r_out[ind_2] = r2_out
        msg_r[it][s + 1] = r_out

      # 右到左更新 L
      for s in range(n - 1, -1, -1):
        ind_1, ind_2, ind_inv = self._stage_indices(s)
        if s == n - 1:
          l1_in = llr_work[ind_1]
          l2_in = llr_work[ind_2]
        else:
          l_in = msg_l[it][s + 1]
          l1_in = l_in[ind_1]
          l2_in = l_in[ind_2]

        if s == 0:
          r1_in = msg_r_in[ind_1]
          r2_in = msg_r_in[ind_2]
        else:
          r_in = msg_r[it][s]
          r1_in = r_in[ind_1]
          r2_in = r_in[ind_2]

        l1_out = self._f_min_sum(l1_in, l2_in + r2_in)
        l2_out = self._f_min_sum(r1_in, l1_in) + l2_in
        l_out = np.empty(N)
        l_out[ind_1] = l1_out
        l_out[ind_2] = l2_out
        msg_l[it][s] = l_out

      # 早停检查
      soft = msg_l[it][0]
      u_hat = np.zeros(N, dtype=int)
      u_hat[~self.frozen_bits] = (soft[~self.frozen_bits] < 0).astype(int)
      x_hat = polar_encode(u_hat)
      hard_ch = (llr_ch < 0).astype(int)
      if np.array_equal(x_hat, hard_ch):
        break

    soft = msg_l[num_iters - 1][0]
    u_hat = np.zeros(N, dtype=int)
    u_hat[~self.frozen_bits] = (soft[~self.frozen_bits] < 0).astype(int)
    return u_hat, num_iters
