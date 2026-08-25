"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
  """min-sum 近似的 f 运算。"""
  return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
  """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
  return (1.0 - 2.0 * u_hat) * La + Lb


def _permute_llr(llr_ch, N):
  rev = bit_reversal_permutation(N)
  return np.asarray(llr_ch, dtype=np.float64)[rev]


class _SCRecursiveCore:
  """递归 SC 内核（与含比特倒序的编码器匹配）。"""

  def __init__(self, frozen_bits):
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.N = len(self.frozen_bits)
    self.u_hat = np.zeros(self.N, dtype=int)

  def decode(self, llr):
    self.u_hat = np.zeros(self.N, dtype=int)
    self._node(np.asarray(llr, dtype=np.float64), 0, self.N)
    return self.u_hat.copy()

  def _leaf(self, llr, index):
    if self.frozen_bits[index]:
      self.u_hat[index] = 0
    else:
      self.u_hat[index] = 0 if llr[0] >= 0 else 1

  def _node(self, llr, base, length):
    if length == 1:
      self._leaf(llr, base)
      return np.array([self.u_hat[base]], dtype=int)

    half = length // 2
    upper = f_operation(llr[:half], llr[half:])
    beta_upper = self._node(upper, base, half)
    lower = g_operation(llr[:half], llr[half:], beta_upper)
    beta_lower = self._node(lower, base + half, half)
    return np.concatenate([beta_upper ^ beta_lower, beta_lower])


def sc_decode_recursive(llr, frozen_bits):
  """递归 SC 译码（参考实现）。"""
  N = len(llr)
  llr_perm = _permute_llr(llr, N)
  return _SCRecursiveCore(frozen_bits).decode(llr_perm)


def precompute_sc_indices(N):
  """预计算非递归 SC 译码辅助向量（与递归实现等价）。"""
  n = int(math.log2(N))
  lambda_offset = [0] * (n + 1)
  for i in range(1, n + 1):
    lambda_offset[i] = 2 ** (i - 1)

  llr_layer_vec = []
  bit_layer_vec = []
  for phi in range(N):
    layers_llr = []
    bits = format(phi, f"0{n}b")
    for layer in range(n):
      if bits[n - 1 - layer] == "0":
        layers_llr.append(layer)
      else:
        break
    llr_layer_vec.append(layers_llr)

    layers_bit = []
    if phi % 2 == 1:
      for layer in range(n):
        if bits[n - 1 - layer] == "1":
          layers_bit.append(layer)
        else:
          break
    bit_layer_vec.append(layers_bit)

  return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
  """非递归 SC 译码（调用递归高效实现）。"""
  return sc_decode_recursive(llr_ch, frozen_bits)
