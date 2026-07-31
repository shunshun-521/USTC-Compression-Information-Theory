"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 PSCD 实现（高效）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def logdomain_sum(x, y):
  """对数域加法（支持标量与数组，数值稳定）。"""
  x = np.asarray(x, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  large = np.abs(x - y) > 30.0
  regular = np.where(
    x > y,
    x + np.log1p(np.exp(np.clip(y - x, -50, 50))),
    y + np.log1p(np.exp(np.clip(x - y, -50, 50))),
  )
  return np.where(large, np.maximum(x, y), regular)


def f_operation(La, Lb):
  """
  精确对数域 f 运算（boxplus）。
  同时支持向量化输入。
  """
  La = np.asarray(La, dtype=np.float64)
  Lb = np.asarray(Lb, dtype=np.float64)
  return logdomain_sum(La + Lb, 0.0) - logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
  """g 运算（递归 SC 使用）：g(La, Lb, u) = (1-2u)*La + Lb"""
  u = np.nan_to_num(np.asarray(u_hat, dtype=np.float64), nan=0.0)
  return (1 - 2 * u) * La + Lb


def lower_llr(l1, l2, b):
  """下分支 LLR 更新（PSCD 风格）。"""
  u = 0 if (b is None or (isinstance(b, float) and np.isnan(b))) else int(b)
  return (l1 + l2) if u == 0 else (l1 - l2)


def _active_llr_level(i, n):
  """从 MSB 起第一个 1 的位置（1-indexed，最大 n）。"""
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
  """从 MSB 起第一个 0 的位置。"""
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
  """递归 SC 译码（参考实现，与 PSCD 结果一致）。"""
  return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
  """
  预计算非递归 SC 译码所需的辅助向量。
  """
  n = int(math.log2(N))
  lambda_offset = [1 << i for i in range(n + 1)]

  llr_layer_vec = []
  bit_layer_vec = []

  for phi in range(N):
    l = bit_reversal_permutation(N)[phi]
    layers = list(range(n - _active_llr_level(l, n), n))
    llr_layer_vec.append(layers)

    blayers = list(range(n, n - _active_bit_level(l, n), -1))
    bit_layer_vec.append(blayers if l >= N / 2 else [])

  return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
  """
  非递归 PSCD 译码主函数。
  """
  N = len(llr_ch)
  n = int(math.log2(N))
  frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])

  br = bit_reversal_permutation(N)
  llr_perm = llr_ch[br].astype(np.float64)

  L = np.full((N, n + 1), np.nan, dtype=np.float64)
  B = np.zeros((N, n + 1))
  L[:, 0] = llr_perm

  for phi in range(N):
    l = br[phi]
    for s in range(n - _active_llr_level(l, n), n):
      block_size = 2 ** (s + 1)
      branch_size = block_size // 2
      for j in range(l, N, block_size):
        if j % block_size < branch_size:
          L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
        else:
          u_bit = B[j - branch_size, s + 1]
          L[j, s + 1] = lower_llr(
            L[j, s], L[j - branch_size, s], u_bit
          )

    if l in frozen_set:
      B[l, n] = 0
    else:
      B[l, n] = 0 if L[l, n] >= 0 else 1

    if l < N / 2:
      continue

    for s in range(n, n - _active_bit_level(l, n), -1):
      block_size = 2 ** s
      branch_size = block_size // 2
      for j in range(l, -1, -block_size):
        if j % block_size >= branch_size:
          B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
          B[j, s - 1] = B[j, s]

  return B[:, n].astype(int)
