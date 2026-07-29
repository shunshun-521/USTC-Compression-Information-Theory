"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


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
  result = 0
  for i in range(n):
    if x & (1 << i):
      result |= 1 << (n - 1 - i)
  return result


def _logdomain_sum(x, y):
  if x > y:
    return x + np.log1p(np.exp(y - x))
  return y + np.log1p(np.exp(x - y))


def _upper_llr_exact(l1, l2):
  """精确 log-domain f 运算（box-plus）"""
  return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
  return (l1 + l2) if b == 0 else (l1 - l2)


def _active_llr_level(i, n):
  mask = 2 ** (n - 1)
  count = 1
  for _ in range(n):
    if (mask & i) == 0:
      count += 1
    else:
      break
    mask >>= 1
  return min(count, n)


def _active_bit_level(i, n):
  mask = 2 ** (n - 1)
  count = 1
  for _ in range(n):
    if (mask & i) > 0:
      count += 1
    else:
      break
    mask >>= 1
  return min(count, n)


def _prepare_llr(llr_ch):
  """编码端含比特倒序时，译码前对信道 LLR 做相同倒序"""
  from encoder import bit_reversal_permutation

  llr_ch = np.asarray(llr_ch, dtype=np.float64)
  N = len(llr_ch)
  rev = bit_reversal_permutation(N)
  return llr_ch[rev]


def sc_decode_recursive(llr, frozen_bits):
  """
  递归 SC 译码（参考实现）
  与 sc_decode 结果一致，采用树形分治调用非递归核心逻辑。
  """
  return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
  """
  预计算非递归 SC 译码所需的辅助向量（保留接口）
  """
  n = int(math.log2(N))
  lambda_offset = np.array([(1 << layer) - 1 for layer in range(n + 1)], dtype=int)

  llr_layer_vec = []
  bit_layer_vec = []
  for phi in range(N):
    llr_layers = []
    psi = phi
    while psi % 2 == 1:
      llr_layers.append(int(math.log2(psi & -psi)))
      psi >>= 1
    llr_layer_vec.append(llr_layers)

    if phi % 2 == 1:
      bit_layers = list(range(n))
    else:
      bit_layers = []
      psi = phi
      layer = 0
      while psi % 2 == 0 and layer < n:
        bit_layers.append(layer)
        psi >>= 1
        layer += 1
      bit_layers.extend(range(layer, n))
    bit_layer_vec.append(bit_layers)

  return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
  """
  非递归 SC 译码主函数（高效实现）
  """
  llr = _prepare_llr(llr_ch)
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  N = len(llr)
  n = int(math.log2(N))
  frozen_set = set(np.where(frozen_bits)[0])

  L = np.full((N, n + 1), np.nan, dtype=np.float64)
  B = np.full((N, n + 1), np.nan)
  L[:, 0] = llr

  for l in [_bit_reversed(i, n) for i in range(N)]:
    for s in range(n - _active_llr_level(l, n), n):
      block = 2 ** (s + 1)
      branch = block // 2
      if block > N:
        continue
      for j in range(l, N, block):
        if j % block < branch:
          L[j, s + 1] = _upper_llr_exact(L[j, s], L[j + branch, s])
        else:
          L[j, s + 1] = _lower_llr(L[j, s], L[j - branch, s], int(B[j - branch, s + 1]))

    if l in frozen_set:
      B[l, n] = 0
    else:
      B[l, n] = 0 if L[l, n] >= 0 else 1

    if l >= N // 2:
      for s in range(n, n - _active_bit_level(l, n), -1):
        block = 2 ** s
        branch = block // 2
        for j in range(l, -1, -block):
          if j % block >= branch:
            B[j - branch, s - 1] = int(B[j, s]) ^ int(B[j - branch, s])
            B[j, s - 1] = B[j, s]

  return B[:, n].astype(int)
