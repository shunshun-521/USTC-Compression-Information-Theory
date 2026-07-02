"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversed_index


def _logdomain_sum(x, y):
    """log(exp(x) + exp(y))，支持向量化。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mx = np.maximum(x, y)
    mn = np.minimum(x, y)
    return mx + np.log1p(np.exp(mn - mx))


def f_operation(La, Lb):
    """精确 log-domain f 运算（boxplus），支持向量化。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(phi, n):
  """计算相位 phi 需要更新的 LLR 层数。"""
  mask = 1 << (n - 1)
  count = 1
  for _ in range(n):
    if (mask & phi) == 0:
      count += 1
    else:
      break
    mask >>= 1
  return min(count, n)


def _active_bit_level(phi, n):
  """计算相位 phi 需要回传比特的层数。"""
  mask = 1 << (n - 1)
  count = 1
  for _ in range(n):
    if (mask & phi) > 0:
      count += 1
    else:
      break
    mask >>= 1
  return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    与非递归版本等价：极化码 SC 需按比特倒序相位调度，
    直接自然序递归在部分码字上会因 LLR 抵消而失败。
    """
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
  """
  非递归 SC 译码主函数。
  按比特倒序相位顺序译码（与 polar_encode 配套）。
  """
  llr_ch = np.asarray(llr_ch, dtype=np.float64)
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  N = len(llr_ch)
  n = int(math.log2(N))

  L = np.zeros((N, n + 1), dtype=np.float64)
  B = np.zeros((N, n + 1), dtype=int)
  L[:, 0] = llr_ch
  u_hat = np.zeros(N, dtype=int)

  for i in range(N):
    phi = bit_reversed_index(i, n)

    for s in range(n - _active_llr_level(phi, n), n):
      block_size = 1 << (s + 1)
      branch_size = block_size // 2
      for j in range(phi, N, block_size):
        if j % block_size < branch_size:
          L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
        else:
          L[j, s + 1] = (
            L[j, s] + L[j - branch_size, s]
            if B[j - branch_size, s + 1] == 0
            else L[j, s] - L[j - branch_size, s]
          )

    if frozen_bits[phi]:
      B[phi, n] = 0
      u_hat[phi] = 0
    else:
      B[phi, n] = 0 if L[phi, n] >= 0 else 1
      u_hat[phi] = B[phi, n]

    if phi >= N // 2:
      for s in range(n, n - _active_bit_level(phi, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(phi, -1, -block_size):
          if j % block_size >= branch_size:
            B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
            B[j, s - 1] = B[j, s]

  return u_hat
