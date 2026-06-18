"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def _bit_reversed(i, n):
  result = 0
  for k in range(n):
    if i & (1 << k):
      result |= 1 << (n - 1 - k)
  return result


def _logdomain_sum(x, y):
  if x > y:
    return x + np.log1p(np.exp(y - x))
  return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
  """对数域 f 运算（向量化）。"""
  La = np.asarray(La, dtype=np.float64)
  Lb = np.asarray(Lb, dtype=np.float64)
  result = np.empty_like(La)
  for idx in range(La.size):
    l1 = La.flat[idx]
    l2 = Lb.flat[idx]
    if np.isinf(l1) and not np.isinf(l2):
      result.flat[idx] = l2
    elif not np.isinf(l1) and np.isinf(l2):
      result.flat[idx] = l1
    elif np.isinf(l1) and np.isinf(l2):
      result.flat[idx] = np.inf
    else:
      result.flat[idx] = _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)
  return result


def g_operation(La, Lb, u_hat):
  """对数域 g 运算（La=top, Lb=bottom）。"""
  La = np.asarray(La, dtype=np.float64)
  Lb = np.asarray(Lb, dtype=np.float64)
  u_hat = np.asarray(u_hat)
  result = np.empty_like(La)
  for idx in range(La.size):
    top = La.flat[idx]
    btm = Lb.flat[idx]
    bit = int(u_hat.flat[idx]) if u_hat.size > 1 else int(u_hat)
    if bit == 0:
      if top == np.inf or btm == np.inf:
        result.flat[idx] = np.inf
      else:
        result.flat[idx] = top + btm
    else:
      result.flat[idx] = btm - top
  return result


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


def sc_decode_recursive(llr, frozen_bits):
  """递归 SC 译码（参考实现）。"""
  N = len(llr)
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  u_hat = np.zeros(N, dtype=int)

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
  """预计算非递归 SC 译码辅助向量。"""
  n = int(math.log2(N))
  lambda_offset = [0] * (n + 1)
  for layer in range(1, n + 1):
    lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer + 1))

  llr_layer_vec = []
  bit_layer_vec = []
  for phi in range(N):
    br_phi = _bit_reversed(phi, n)
    start = n - _active_llr_level(br_phi, n)
    llr_layer_vec.append(list(range(start, n)))
    bit_start = n - _active_bit_level(br_phi, n) + 1
    bit_layer_vec.append(list(range(n, bit_start - 1, -1)))
  return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs(L, B, l, n, N):
  for s in range(n - _active_llr_level(l, n), n):
    block_size = 1 << (s + 1)
    branch_size = block_size >> 1
    for j in range(l, N, block_size):
      if j % block_size < branch_size:
        L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
      else:
        L[j, s + 1] = g_operation(
          L[j - branch_size, s],
          L[j, s],
          B[j - branch_size, s + 1],
        )


def _update_bits(B, l, n, N):
  if l < N // 2:
    return
  for s in range(n, n - _active_bit_level(l, n), -1):
    block_size = 1 << s
    branch_size = block_size >> 1
    for j in range(l, -1, -block_size):
      if j % block_size >= branch_size:
        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
        B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
  """非递归 Permuted SC 译码主函数。"""
  llr_ch = np.asarray(llr_ch, dtype=np.float64)
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  N = len(llr_ch)
  n = int(math.log2(N))

  L = np.full((N, n + 1), np.nan, dtype=np.float64)
  B = np.zeros((N, n + 1), dtype=int)
  L[:, 0] = llr_ch
  frozen_set = set(np.where(frozen_bits)[0])

  for phi in range(N):
    l = _bit_reversed(phi, n)
    _update_llrs(L, B, l, n, N)
    if l in frozen_set:
      B[l, n] = 0
    else:
      B[l, n] = 0 if L[l, n] >= 0 else 1
    _update_bits(B, l, n, N)

  return B[:, n].astype(int)
