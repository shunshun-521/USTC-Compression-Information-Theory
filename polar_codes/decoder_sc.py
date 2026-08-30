"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted SCD）
"""
import math

import numpy as np


def bit_reversed(x, n):
  """比特倒序。"""
  if np.isscalar(x):
    result = 0
    for i in range(n):
      if x & (1 << i):
        result |= 1 << (n - 1 - i)
    return result
  return np.array([bit_reversed(int(v), n) for v in x])


def _logdomain_sum(x, y):
  if x > y:
    return x + np.log1p(np.exp(y - x))
  return y + np.log1p(np.exp(x - y))


def _logdomain_diff(x, y):
  if x > y:
    return x + np.log1p(-np.exp(y - x))
  return y + np.log1p(-np.exp(x - y))


def f_operation(La, Lb):
  """
  f 运算（log-domain box-plus，向量化）。
  """
  La = np.asarray(La, dtype=np.float64)
  Lb = np.asarray(Lb, dtype=np.float64)
  return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
  """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
  La = np.asarray(La, dtype=np.float64)
  Lb = np.asarray(Lb, dtype=np.float64)
  u_hat = np.asarray(u_hat)
  return np.where(u_hat == 0, La + Lb, La - Lb)


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


def _upper_llr(l1, l2):
  if np.isinf(l1) and not np.isinf(l2):
    return l2
  if not np.isinf(l1) and np.isinf(l2):
    return l1
  if np.isinf(l1) and np.isinf(l2):
    return np.inf
  return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
  if b == 0:
    if np.isinf(l1) or np.isinf(l2):
      return np.inf
    return l1 + l2
  return l1 - l2


def sc_decode_recursive(llr, frozen_bits):
  """递归 SC 译码（参考实现）。"""
  llr = np.asarray(llr, dtype=np.float64)
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  N = len(llr)
  u_hat = np.zeros(N, dtype=np.int8)

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

    u_left = u_hat[bit_offset : bit_offset + half]
    llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
    decode_node(llr_right, bit_offset + half)

  decode_node(llr, 0)
  return u_hat


def precompute_sc_indices(N):
  """预计算非递归 SC 译码辅助信息（兼容接口）。"""
  n = int(math.log2(N))
  lambda_offset = [1 << i for i in range(n + 1)]
  llr_layer_vec = [list(range(n - _active_llr_level(bit_reversed(phi, n), n), n)) for phi in range(N)]
  bit_layer_vec = [list(range(n, n - _active_bit_level(bit_reversed(phi, n), n), -1)) for phi in range(N)]
  return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
  """
  非递归 Permuted SC 译码（高效实现）。
  信道 LLR 为编码后码字顺序；内部做比特倒序以匹配蝶形因子图。
  """
  llr_ch = np.asarray(llr_ch, dtype=np.float64)
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  N = len(llr_ch)
  n = int(math.log2(N))
  frozen_set = set(np.where(frozen_bits)[0])
  rev_perm = np.array([bit_reversed(i, n) for i in range(N)], dtype=int)
  llr_dec = llr_ch[rev_perm]

  L = np.full((N, n + 1), np.nan, dtype=np.float64)
  B = np.full((N, n + 1), np.nan)
  L[:, 0] = llr_dec

  for phi in range(N):
    l = bit_reversed(phi, n)
    for s in range(n - _active_llr_level(l, n), n):
      block_size = 1 << (s + 1)
      branch_size = block_size // 2
      for j in range(l, N, block_size):
        if j % block_size < branch_size:
          L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
        else:
          L[j, s + 1] = _lower_llr(
            L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
          )

    if l in frozen_set:
      B[l, n] = 0
    else:
      B[l, n] = 0 if L[l, n] >= 0 else 1

    if l >= N // 2:
      for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
          if j % block_size >= branch_size:
            B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
            B[j, s - 1] = B[j, s]

  return B[:, n].astype(np.int8)


def verify_sc_decoder(N=64, K=32, num_frames=100, eb_n0_db=12.0):
  """在极低噪声下验证 SC 译码正确性。"""
  from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
  from construction import ga_construction
  from encoder import polar_encode

  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=bool)
  frozen_bits[info_idx] = False
  rate = K / N
  sigma = eb_n0_to_sigma(eb_n0_db, rate)
  rng = np.random.default_rng(0)

  for _ in range(num_frames):
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat = sc_decode(llr, frozen_bits)
    if not np.array_equal(u_hat[info_idx], u[info_idx]):
      return False
  return True
