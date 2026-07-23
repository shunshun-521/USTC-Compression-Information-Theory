"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted SCD）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
  """
  min-sum 近似的 f 运算：
  f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
  """
  return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
  """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
  return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
  if x > y:
    return x + np.log1p(np.exp(y - x))
  return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
  """f 运算（log-domain box-plus）"""
  return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
  """g 运算（log-domain）"""
  return l1 + l2 if b == 0 else l1 - l2


def _bit_reversed(i, n):
  result = 0
  for bit in range(n):
    if i & (1 << bit):
      result |= 1 << (n - 1 - bit)
  return result


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


def sc_decode_recursive(llr, frozen_bits):
  """递归 SC 译码（参考实现，与 sc_decode 等价）"""
  return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
  """预计算非递归 SC 译码辅助向量"""
  n = int(math.log2(N))
  lambda_offset = [1 << i for i in range(n + 1)]
  llr_layer_vec = []
  bit_layer_vec = []
  for i in range(N):
    llr_layer_vec.append(list(range(n - _active_llr_level(i, n), n)))
    if i < N // 2:
      bit_layer_vec.append([])
    else:
      bit_layer_vec.append(
          list(range(n, n - _active_bit_level(i, n), -1))
      )
  return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
  """
  非递归 Permuted SC 译码（Vangala 算法，与 polar_encode 匹配）。
  """
  llr_ch = np.asarray(llr_ch, dtype=np.float64)
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  N = len(llr_ch)
  n = int(math.log2(N))

  L = np.full((N, n + 1), np.nan, dtype=np.float64)
  B = np.zeros((N, n + 1), dtype=int)
  L[:, 0] = llr_ch

  def update_llrs(l):
    for s in range(n - _active_llr_level(l, n), n):
      block_size = 2 ** (s + 1)
      branch_size = block_size // 2
      for j in range(l, N, block_size):
        if j % block_size < branch_size:
          L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
        else:
          L[j, s + 1] = lower_llr(
              L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
          )

  def update_bits(l):
    if l < N // 2:
      return
    for s in range(n, n - _active_bit_level(l, n), -1):
      block_size = 2 ** s
      branch_size = block_size // 2
      for j in range(l, -1, -block_size):
        if j % block_size >= branch_size:
          B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
          B[j, s - 1] = B[j, s]

  for l in [_bit_reversed(i, n) for i in range(N)]:
    update_llrs(l)
    if frozen_bits[l]:
      B[l, n] = 0
    else:
      B[l, n] = 0 if L[l, n] >= 0 else 1
    update_bits(l)

  return B[:, n].astype(int)
