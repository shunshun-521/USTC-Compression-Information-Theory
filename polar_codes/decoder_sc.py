"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Permuted SCD，高效实现）
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
  return (1 - 2 * u_hat) * La + Lb


def _bit_reversed_index(i, n):
  """单索引比特倒序。"""
  result = 0
  for bit in range(n):
    if i & (1 << bit):
      result |= 1 << (n - 1 - bit)
  return result


def _active_llr_level(i, n):
  """从最高位起第一个 0 的位置计数（Permuted SCD）。"""
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
  """从最高位起第一个 1 的位置计数（Permuted SCD）。"""
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
  """Permuted SCD 的 LLR 更新。"""
  for s in range(n - _active_llr_level(l, n), n):
    block_size = 1 << (s + 1)
    branch_size = block_size // 2
    for j in range(l, L.shape[0], block_size):
      if j % block_size < branch_size:
        top_llr = L[j, s]
        btm_llr = L[j + branch_size, s]
        L[j, s + 1] = f_operation(top_llr, btm_llr)
      else:
        btm_llr = L[j, s]
        top_llr = L[j - branch_size, s]
        top_bit = B[j - branch_size, s + 1]
        L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)


def _update_bits(B, l, n):
  """Permuted SCD 的比特回传。"""
  if l < B.shape[0] // 2:
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
    llr_left = f_operation(llr_node[0::2], llr_node[1::2])
    for i in range(half):
      decode_node(llr_left[i : i + 1], bit_offset + i)

    u_left = u_hat[bit_offset : bit_offset + half]
    llr_right = g_operation(llr_node[0::2], llr_node[1::2], u_left)
    for i in range(half):
      decode_node(llr_right[i : i + 1], bit_offset + half + i)

  decode_node(llr, 0)
  return u_hat


def precompute_sc_indices(N):
  """
  预计算非递归 SC 译码所需的辅助向量。
  返回：lambda_offset, llr_layer_vec, bit_layer_vec
  """
  n = int(math.log2(N))
  lambda_offset = [(1 << layer) - 1 for layer in range(n + 1)]

  llr_layer_vec = []
  bit_layer_vec = []
  for phi in range(N):
    l = _bit_reversed_index(phi, n)
    llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
    if l >= N // 2:
      bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    else:
      bit_layer_vec.append([])

  return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
  """
  非递归 Permuted SCD 译码主函数。
  信道 LLR 按自然顺序输入（与编码后的码字比特一一对应）。
  """
  N = len(llr_ch)
  n = int(math.log2(N))
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  frozen_set = set(np.where(frozen_bits)[0])

  L = np.full((N, n + 1), np.nan, dtype=np.float64)
  B = np.full((N, n + 1), np.nan, dtype=np.float64)
  rev = bit_reversal_permutation(N)
  L[:, 0] = np.asarray(llr_ch, dtype=np.float64)[rev]

  u_hat = np.zeros(N, dtype=int)
  decode_order = [_bit_reversed_index(i, n) for i in range(N)]

  for l in decode_order:
    _update_llrs(L, B, l, n)
    if l in frozen_set:
      u_hat[l] = 0
      B[l, n] = 0
    else:
      u_hat[l] = 0 if L[l, n] >= 0 else 1
      B[l, n] = u_hat[l]
    _update_bits(B, l, n)

  return u_hat


def permute_channel_llr(llr_ch):
  """将信道 LLR 按比特倒序置换（与 Permuted SCD 一致）。"""
  N = len(llr_ch)
  rev = bit_reversal_permutation(N)
  return np.asarray(llr_ch, dtype=np.float64)[rev]
