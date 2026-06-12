"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
  """
  min-sum 近似的 f 运算：
  f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
  支持向量化（La, Lb 为同形状 numpy 数组）
  """
  return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
  """
  g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
  """
  return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
  """
  预计算非递归 SC 译码所需的三个辅助向量。
  """
  n = int(math.log2(N))
  lambda_offset = [1 << i for i in range(n + 1)]

  llr_layer_vec = []
  bit_layer_vec = []

  for phi in range(N):
    llr_layers = []
    temp = phi
    layer = 0
    while layer < n:
      llr_layers.append(layer)
      if (temp & 1) == 0:
        break
      temp >>= 1
      layer += 1
    llr_layer_vec.append(llr_layers)

    bit_layers = []
    temp = phi
    layer = 0
    while layer < n:
      if (temp & 1) == 1:
        break
      bit_layers.append(layer)
      temp >>= 1
      layer += 1
    bit_layer_vec.append(bit_layers)

  return lambda_offset, llr_layer_vec, bit_layer_vec


def _active_llr_level(i, n):
  """自 MSB 起第一个 1 之前需要更新的 LLR 层数。"""
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
  """自 MSB 起第一个 0 之前需要回传的比特层数。"""
  mask = 1 << (n - 1)
  count = 1
  for _ in range(n):
    if (mask & i) > 0:
      count += 1
      mask >>= 1
    else:
      break
  return min(count, n)


def _update_llr(L, B, l, n, N):
  """沿比特 l 的路径自信道列向判决列更新 LLR。"""
  start_s = n - _active_llr_level(l, n)
  for s in range(start_s, n):
    block = 1 << (s + 1)
    branch = block >> 1
    j = l
    while j < N:
      if (j % block) < branch:
        L[j, s + 1] = f_operation(L[j, s], L[j + branch, s])
      else:
        L[j, s + 1] = g_operation(
          L[j - branch, s], L[j, s], B[j - branch, s + 1]
        )
      j += block


def _update_bits(B, l, n, N):
  """沿比特 l 的路径回传硬判决。"""
  if l < N // 2:
    return
  end_s = n - _active_bit_level(l, n)
  for s in range(n, end_s, -1):
    block = 1 << s
    branch = block >> 1
    j = l
    while j >= 0:
      if (j % block) >= branch:
        B[j - branch, s - 1] = B[j, s] ^ B[j - branch, s]
        B[j, s - 1] = B[j, s]
      j -= block


def _sc_decode_core(llr_ch, frozen_bits):
  """
  置换 SC 译码核心（Permuted SC）。
  编码器 x = u * F^{⊗n} 需按比特倒序调度译码。
  """
  N = len(llr_ch)
  n = int(math.log2(N))
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  br = bit_reversal_permutation(N)

  L = np.zeros((N, n + 1), dtype=np.float64)
  B = np.zeros((N, n + 1), dtype=np.int32)
  L[:, 0] = llr_ch

  u_hat = np.zeros(N, dtype=int)

  for phi in range(N):
    l = br[phi]
    _update_llr(L, B, l, n, N)

    if frozen_bits[l]:
      B[l, n] = 0
    else:
      B[l, n] = 0 if L[l, n] >= 0 else 1

    u_hat[l] = B[l, n]
    _update_bits(B, l, n, N)

  return u_hat


def sc_decode(llr_ch, frozen_bits):
  """非递归 SC 译码主函数。"""
  return _sc_decode_core(llr_ch, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
  """
  递归 SC 译码（参考实现）。
  对每一比特递归调用 LLR/比特更新子过程，与 sc_decode 等价。
  """
  N = len(llr)
  n = int(math.log2(N))
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  br = bit_reversal_permutation(N)

  L = np.zeros((N, n + 1), dtype=np.float64)
  B = np.zeros((N, n + 1), dtype=np.int32)
  L[:, 0] = llr
  u_hat = np.zeros(N, dtype=int)

  def decode_bit(phi):
    l = br[phi]
    _update_llr_recursive(L, B, l, n, N)
    if frozen_bits[l]:
      B[l, n] = 0
    else:
      B[l, n] = 0 if L[l, n] >= 0 else 1
    u_hat[l] = B[l, n]
    _update_bits_recursive(B, l, n, N)

  for phi in range(N):
    decode_bit(phi)

  return u_hat


def _update_llr_recursive(L, B, l, n, N):
  """递归形式沿路径更新 LLR。"""

  def recurse(s):
    if s >= n:
      return
    block = 1 << (s + 1)
    branch = block >> 1
    j = l
    while j < N:
      if (j % block) < branch:
        L[j, s + 1] = f_operation(L[j, s], L[j + branch, s])
      else:
        L[j, s + 1] = g_operation(
          L[j - branch, s], L[j, s], B[j - branch, s + 1]
        )
      j += block
    recurse(s + 1)

  start_s = n - _active_llr_level(l, n)
  recurse(start_s)


def _update_bits_recursive(B, l, n, N):
  """递归形式回传比特。"""

  def recurse(s):
    if s < n - _active_bit_level(l, n):
      return
    block = 1 << s
    branch = block >> 1
    j = l
    while j >= 0:
      if (j % block) >= branch:
        B[j - branch, s - 1] = B[j, s] ^ B[j - branch, s]
        B[j, s - 1] = B[j, s]
      j -= block
    recurse(s - 1)

  if l < N // 2:
    return
  recurse(n)
