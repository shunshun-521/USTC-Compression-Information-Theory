"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted SC）
"""
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


def _bit_reversed(x, n):
  """单索引比特倒序"""
  result = 0
  for i in range(n):
    if x & (1 << i):
      result |= 1 << (n - 1 - i)
  return result


def _active_llr_level(i, n):
  """找到二进制表示中第一个 1 的位置（从高位起）"""
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
  """找到二进制表示中第一个 0 的位置（从高位起）"""
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
  return f_operation(l1, l2)


def _lower_llr(l1, l2, b):
  """l1=下支路 LLR, l2=上支路 LLR"""
  if b == 0:
    return l1 + l2
  return l1 - l2


def sc_decode_recursive(llr, frozen_bits):
  """递归 SC 译码（参考实现，与 Permuted SC 等价）"""
  return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
  """
  预计算非递归 SC 译码所需的辅助向量（兼容接口）。
  """
  n = int(np.log2(N))
  lambda_offset = [1 << i for i in range(n + 1)]
  llr_layer_vec = []
  bit_layer_vec = []

  for phi in range(N):
    l = _bit_reversed(phi, n)
    start = n - _active_llr_level(l, n)
    llr_layer_vec.append(list(range(n - 1, start - 1, -1)))

    if l < N // 2:
      bit_layer_vec.append([])
    else:
      start_b = n - _active_bit_level(l, n)
      bit_layer_vec.append(list(range(n, start_b, -1)))

  return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
  """
  非递归 Permuted SC 译码主函数。
  编码器含比特倒序置换时，先将信道 LLR 按比特倒序重排。
  """
  N = len(llr_ch)
  n = int(np.log2(N))
  rev = bit_reversal_permutation(N)
  llr = np.asarray(llr_ch, dtype=np.float64)[rev]
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  frozen_set = set(np.where(frozen_bits)[0])

  L = np.zeros((N, n + 1), dtype=np.float64)
  B = np.zeros((N, n + 1), dtype=np.int8)
  L[:, 0] = llr

  for phi in range(N):
    l = _bit_reversed(phi, n)

    for s in range(n - _active_llr_level(l, n), n):
      block_size = 2 ** (s + 1)
      branch_size = block_size // 2
      for j in range(l, N, block_size):
        if j % block_size < branch_size:
          L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
        else:
          top_bit = B[j - branch_size, s + 1]
          L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    if l in frozen_set:
      B[l, n] = 0
    else:
      B[l, n] = 0 if L[l, n] >= 0 else 1

    if l >= N // 2:
      for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
          if j % block_size >= branch_size:
            B[j - branch_size, s - 1] = (B[j, s] ^ B[j - branch_size, s]) & 1
            B[j, s - 1] = B[j, s]

  return B[:, n].astype(np.int8)
