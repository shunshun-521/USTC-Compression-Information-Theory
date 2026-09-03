"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 Permuted SCD（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _logdomain_sum(x, y):
    """log(exp(x) + exp(y))，数值稳定"""
    if np.isscalar(x):
        if x == np.inf and y != np.inf:
            return y
        if y == np.inf and x != np.inf:
            return x
        if x == np.inf and y == np.inf:
            return np.inf
        if x == -np.inf and y != -np.inf:
            return y
        if y == -np.inf and x != -np.inf:
            return x
        if x == -np.inf and y == -np.inf:
            return -np.inf
        if abs(x - y) >= 30:
            return max(x, y)
        return x + np.log1p(np.exp(y - x))
    # vectorized fallback
    return np.vectorize(_logdomain_sum, otypes=[float])(x, y)


def _upper_llr(l1, l2):
    """精确 log-domain f 运算"""
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l2 == np.inf and l1 != np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    """精确 log-domain g 运算"""
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    return l1 - l2


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(np.asarray(llr, dtype=np.float64), 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（Permuted SCD 索引）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))

        if l < N // 2:
            bit_layer_vec.append([])
        else:
            start_b = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, start_b, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCDState:
  """Permuted SCD 内部状态"""

  def __init__(self, N, n, llr_ch):
      self.N = N
      self.n = n
      self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
      self.B = np.full((N, n + 1), np.nan)
      self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)

  def update_llrs(self, l):
      for s in range(self.n - _active_llr_level(l, self.n), self.n):
          block_size = 2 ** (s + 1)
          branch_size = block_size // 2
          for j in range(l, self.N, block_size):
              if j % block_size < branch_size:
                  self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + branch_size, s])
              else:
                  self.L[j, s + 1] = _lower_llr(
                      self.L[j, s],
                      self.L[j - branch_size, s],
                      int(self.B[j - branch_size, s + 1]),
                  )

  def update_bits(self, l):
      if l < self.N / 2:
          return
      for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
          block_size = 2 ** s
          branch_size = block_size // 2
          for j in range(l, -1, -block_size):
              if j % block_size >= branch_size:
                  self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                      self.B[j - branch_size, s]
                  )
                  self.B[j, s - 1] = self.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SCD 译码主函数。
    信道 LLR 在入口处做比特倒序，与含 B_N 的编码器配套。
    """
    from encoder import bit_reversal_permutation

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    llr_ch = llr_ch[bit_reversal_permutation(len(llr_ch))]
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    state = _SCDState(N, n, llr_ch)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = _bit_reversed(phi, n)
        state.update_llrs(l)
        if frozen_bits[l]:
            state.B[l, n] = 0
        else:
            state.B[l, n] = 0 if state.L[l, n] >= 0 else 1
        u_hat[l] = int(state.B[l, n])
        state.update_bits(l)

    return u_hat
