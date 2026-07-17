"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def _logdomain_sum(x, y):
  if x > y:
    return x + np.log1p(np.exp(y - x))
  return y + np.log1p(np.exp(x - y))


def _bit_reversed_index(i, n):
  result = 0
  for bit in range(n):
    if i & (1 << bit):
      result |= 1 << (n - 1 - bit)
  return result


def f_operation(La, Lb):
  """精确 log-domain box-plus f 运算。"""
  La = np.asarray(La, dtype=np.float64)
  Lb = np.asarray(Lb, dtype=np.float64)
  if La.ndim == 0 and Lb.ndim == 0:
    return float(_logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb))
  return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
  """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
  La = np.asarray(La, dtype=np.float64)
  Lb = np.asarray(Lb, dtype=np.float64)
  u_hat = np.asarray(u_hat)
  return np.where(u_hat == 0, La + Lb, La - Lb)


def _active_llr_level(i, n):
  mask = 1 << (n - 1)
  count = 1
  for _ in range(n):
    if (mask & i) == 0:
      count += 1
      mask >>= 1
    else:
      break
  return min(count, n)


def _active_bit_level(i, n):
  mask = 1 << (n - 1)
  count = 1
  for _ in range(n):
    if (mask & i) > 0:
      count += 1
      mask >>= 1
    else:
      break
  return min(count, n)


def _update_llrs(L, B, l, n):
  for s in range(n - _active_llr_level(l, n), n):
    block_size = 1 << (s + 1)
    branch_size = block_size // 2
    for j in range(l, L.shape[0], block_size):
      if j % block_size < branch_size:
        L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
      else:
        L[j, s + 1] = g_operation(
          L[j, s],
          L[j - branch_size, s],
          B[j - branch_size, s + 1],
        )


def _update_bits(B, l, n, N):
  if l < N / 2:
    return
  for s in range(n, n - _active_bit_level(l, n), -1):
    block_size = 1 << s
    branch_size = block_size // 2
    for j in range(l, -1, -block_size):
      if j % block_size >= branch_size:
        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
        B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
  """递归 SC 译码（参考实现）。"""
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  N = len(llr)
  n = int(np.log2(N))
  order = [_bit_reversed_index(i, n) for i in range(N)]

  L = np.full((N, n + 1), np.nan, dtype=np.float64)
  B = np.full((N, n + 1), np.nan)
  rev = bit_reversal_permutation(N)
  L[:, 0] = llr[rev]

  for l in order:
    _update_llrs(L, B, l, n)
    if frozen_bits[l]:
      B[l, n] = 0
    else:
      B[l, n] = 0 if L[l, n] >= 0 else 1
    _update_bits(B, l, n, N)

  return B[:, n].astype(int)


def precompute_sc_indices(N):
  """预计算非递归 SC 译码所需的辅助向量。"""
  n = int(np.log2(N))
  lambda_offset = [1 << i for i in range(n + 1)]
  llr_layer_vec = []
  bit_layer_vec = []

  for phi in range(N):
    llr_layer_vec.append(list(range(n - _active_llr_level(phi, n), n)))
    bit_layer_vec.append(list(range(n, n - _active_bit_level(phi, n), -1)))

  return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
  """非递归 SC 译码主函数。"""
  llr_ch = np.asarray(llr_ch, dtype=np.float64)
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  N = len(llr_ch)
  n = int(np.log2(N))

  L = np.full((N, n + 1), np.nan, dtype=np.float64)
  B = np.full((N, n + 1), np.nan)
  rev = bit_reversal_permutation(N)
  L[:, 0] = llr_ch[rev]

  order = [_bit_reversed_index(i, n) for i in range(N)]
  for l in order:
    _update_llrs(L, B, l, n)
    if frozen_bits[l]:
      B[l, n] = 0
    else:
      B[l, n] = 0 if L[l, n] >= 0 else 1
    _update_bits(B, l, n, N)

  return B[:, n].astype(int)
