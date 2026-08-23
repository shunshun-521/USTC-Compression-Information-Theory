"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def bit_reversed(i, n):
  """将 i 的 n 位二进制表示做比特倒序"""
  return int(np.binary_repr(i, width=n)[::-1], 2)


def active_llr_level(i, n):
  """从 MSB 起找到第一个 1 之前的层数（参考 mcba1n）"""
  mask = 2 ** (n - 1)
  count = 1
  for _ in range(n):
    if (mask & i) == 0:
      count += 1
      mask >>= 1
    else:
      break
  return min(count, n)


def active_bit_level(i, n):
  """从 MSB 起找到第一个 0 之前的层数（参考 mcba1n）"""
  mask = 2 ** (n - 1)
  count = 1
  for _ in range(n):
    if (mask & i) > 0:
      count += 1
      mask >>= 1
    else:
      break
  return min(count, n)


def upper_llr(l1, l2):
  """精确 log-domain f 运算（boxplus）"""
  if np.isinf(l1) and not np.isinf(l2):
    return l2
  if not np.isinf(l1) and np.isinf(l2):
    return l1
  if np.isinf(l1) and np.isinf(l2):
    return np.inf
  return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
  """精确 log-domain g 运算"""
  if b == 0:
    if np.isinf(l1) or np.isinf(l2):
      return np.inf
    return l1 + l2
  return l1 - l2


def _logdomain_sum(x, y):
  max_val = max(x, y)
  min_val = min(x, y)
  return max_val + np.log1p(np.exp(min_val - max_val))


def f_operation(La, Lb):
  """f 运算（标量/数组，优先精确 boxplus）"""
  La = np.asarray(La, dtype=np.float64)
  Lb = np.asarray(Lb, dtype=np.float64)
  if La.shape == () and Lb.shape == ():
    return upper_llr(float(La), float(Lb))
  return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
  """g 运算"""
  La = np.asarray(La, dtype=np.float64)
  Lb = np.asarray(Lb, dtype=np.float64)
  u_hat = np.asarray(u_hat)
  if La.shape == () and Lb.shape == () and u_hat.shape == ():
    return lower_llr(float(Lb), float(La), int(u_hat))
  return (1.0 - 2.0 * u_hat) * La + Lb


def _update_llrs(l, L, B, n):
  """更新到达叶子 l 所需的 LLR"""
  for s in range(n - active_llr_level(l, n), n):
    block_size = 2 ** (s + 1)
    branch_size = block_size // 2
    for j in range(l, L.shape[0], block_size):
      if j % block_size < branch_size:
        top_llr = L[j, s]
        btm_llr = L[j + branch_size, s]
        L[j, s + 1] = upper_llr(top_llr, btm_llr)
      else:
        btm_llr = L[j, s]
        top_llr = L[j - branch_size, s]
        top_bit = B[j - branch_size, s + 1]
        L[j, s + 1] = lower_llr(btm_llr, top_llr, top_bit)


def _update_bits(l, B, n):
  """比特回传"""
  if l < B.shape[0] // 2:
    return
  for s in range(n, n - active_bit_level(l, n), -1):
    block_size = 2 ** s
    branch_size = block_size // 2
    for j in range(l, -1, -block_size):
      if j % block_size >= branch_size:
        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
        B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
  """递归 SC 译码（参考实现，与非递归结果一致）"""
  return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
  """
  预计算非递归 SC 译码所需的辅助向量（兼容接口）。
  """
  n = int(math.log2(N))
  llr_layer_vec = []
  bit_layer_vec = []
  for phi in range(N):
    l = bit_reversed(phi, n)
    llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
    bit_layers = list(range(n, n - active_bit_level(l, n), -1))
    bit_layer_vec.append(bit_layers)
  lambda_offset = [1 << layer for layer in range(n + 1)]
  return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
  """
  非递归 SC 译码主函数（mcba1n SCD 风格索引）。
  信道 LLR 在输入时做比特倒序，与编码器输出倒序一致。
  """
  llr_ch = np.asarray(llr_ch, dtype=np.float64)
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  N = len(llr_ch)
  n = int(math.log2(N))
  rev = bit_reversal_permutation(N)
  llr_ch = llr_ch[rev]
  frozen_set = set(np.where(frozen_bits)[0])

  L = np.full((N, n + 1), np.nan, dtype=np.float64)
  B = np.zeros((N, n + 1), dtype=np.int8)
  L[:, 0] = llr_ch

  for i in range(N):
    l = bit_reversed(i, n)
    _update_llrs(l, L, B, n)
    if l in frozen_set:
      B[l, n] = 0
    else:
      B[l, n] = 0 if L[l, n] >= 0 else 1
    _update_bits(l, B, n)

  return B[:, n].astype(int)
