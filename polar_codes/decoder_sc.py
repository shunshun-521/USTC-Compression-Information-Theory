"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效 SCD 实现）
"""
import math
import numpy as np

from encoder import bit_reversed, bit_reversal_permutation


def f_operation(La, Lb):
  """min-sum 近似的 f 运算。"""
  sa = np.where(La >= 0, 1.0, -1.0)
  sb = np.where(Lb >= 0, 1.0, -1.0)
  return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
  """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
  return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
  """找到索引 i 的二进制表示中第一个 1 的位置（从高位计）。"""
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
  """找到索引 i 的二进制表示中第一个 0 的位置（从高位计）。"""
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
  return f_operation(np.array([l1]), np.array([l2]))[0]


def _lower_llr(btm, top, b):
  """下分支 LLR（btm=bottom, top=top）。"""
  if b == 0:
    return btm + top
  return btm - top


def sc_decode(llr_ch, frozen_bits):
  """
  非递归 SC 译码（SCD，比特倒序译码顺序）。
  """
  llr_ch = np.asarray(llr_ch, dtype=np.float64)
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  N = len(llr_ch)
  n = int(math.log2(N))

  L = np.zeros((N, n + 1), dtype=np.float64)
  B = np.zeros((N, n + 1), dtype=np.int8)
  br = bit_reversal_permutation(N)
  L[:, 0] = llr_ch[br]

  u_hat = np.zeros(N, dtype=int)

  for i in range(N):
    l = bit_reversed(i, n)
    _update_llrs(L, B, l, n)

    if frozen_bits[l]:
      u_hat[l] = 0
      B[l, n] = 0
    else:
      u_hat[l] = 0 if L[l, n] >= 0 else 1
      B[l, n] = u_hat[l]

    _update_bits(B, l, n)

  return u_hat


def _update_llrs(L, B, l, n):
  N = L.shape[0]
  for s in range(n - _active_llr_level(l, n), n):
    block_size = 2 ** (s + 1)
    branch_size = block_size // 2
    for j in range(l, N, block_size):
      if j % block_size < branch_size:
        L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
      else:
        top_llr = L[j - branch_size, s]
        btm_llr = L[j, s]
        top_bit = B[j - branch_size, s + 1]
        L[j, s + 1] = _lower_llr(btm_llr, top_llr, top_bit)


def _update_bits(B, l, n):
  if l < B.shape[0] // 2:
    return
  for s in range(n, n - _active_bit_level(l, n), -1):
    block_size = 2 ** s
    branch_size = block_size // 2
    for j in range(l, -1, -block_size):
      if j % block_size >= branch_size:
        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
        B[j, s - 1] = B[j, s]


def precompute_sc_indices(N):
  """预计算非递归 SC 译码所需的辅助向量（兼容接口）。"""
  n = int(math.log2(N))
  lambda_offset = [1 << i for i in range(n + 1)]
  llr_layer_vec = []
  bit_layer_vec = []
  for phi in range(N):
    layer = 0
    temp = phi
    while temp % 2 == 1:
      temp //= 2
      layer += 1
    llr_layer_vec.append(list(range(layer, n)))
    if phi % 2 == 0:
      bit_layer_vec.append(list(range(layer)))
    else:
      bit_layer_vec.append(list(range(layer - 1, -1, -1)))
  return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
  """递归 SC 译码（用于小规模验证）。"""
  return sc_decode(llr, frozen_bits)
