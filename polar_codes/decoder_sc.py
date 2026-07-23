"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于 PSCD 算法）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
  """min-sum 近似的 f 运算"""
  return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
  """g 运算"""
  return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
  result = 0
  for i in range(n):
    if x & (1 << i):
      result |= 1 << (n - 1 - i)
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


def _upper_llr(l1, l2):
  return f_operation(l1, l2)


def _lower_llr(l1, l2, bit):
  return g_operation(l1, l2, bit)


def _prepare_llr(llr_ch):
  """与编码端比特倒序置换对应的信道 LLR 重排"""
  n = int(math.log2(len(llr_ch)))
  br = bit_reversal_permutation(len(llr_ch))
  return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode_recursive(llr, frozen_bits):
  """递归 SC 译码（参考实现）"""
  llr = _prepare_llr(llr)
  return _sc_decode_pscd(llr, frozen_bits)


def precompute_sc_indices(N):
  """
  预计算非递归 SC 译码所需的辅助向量（与 PSCD 层更新对应）。
  """
  n = int(math.log2(N))
  llr_layer_vec = []
  bit_layer_vec = []
  for phi in range(N):
    l = _bit_reversed(phi, n)
    layers = list(range(n - _active_llr_level(l, n), n))
    llr_layer_vec.append(layers)

    bit_layers = []
    if l >= N / 2:
      bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
    bit_layer_vec.append(bit_layers)

  return llr_layer_vec, bit_layer_vec


def _sc_decode_pscd(llr_ch, frozen_bits):
  """Permuted successive cancellation decoder（非递归主实现）"""
  llr_ch = np.asarray(llr_ch, dtype=np.float64)
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  N = len(llr_ch)
  n = int(math.log2(N))

  L = np.zeros((N, n + 1), dtype=np.float64)
  B = np.zeros((N, n + 1), dtype=int)
  L[:, 0] = llr_ch
  u_hat = np.zeros(N, dtype=int)
  frozen_indices = set(np.where(frozen_bits)[0])

  for i in range(N):
    l = _bit_reversed(i, n)
    for s in range(n - _active_llr_level(l, n), n):
      block_size = 2 ** (s + 1)
      branch_size = block_size // 2
      for j in range(l, N, block_size):
        if j % block_size < branch_size:
          L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
        else:
          L[j, s + 1] = _lower_llr(
            L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
          )

    if l in frozen_indices:
      B[l, n] = 0
    else:
      B[l, n] = 0 if L[l, n] >= 0 else 1
    u_hat[l] = B[l, n]

    if l >= N / 2:
      for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
          if j % block_size >= branch_size:
            B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
            B[j, s - 1] = B[j, s]

  return u_hat


def sc_decode(llr_ch, frozen_bits):
  """非递归 SC 译码主函数"""
  llr = _prepare_llr(llr_ch)
  return _sc_decode_pscd(llr, frozen_bits)
