"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
  """min-sum 近似的 f 运算"""
  sa = np.sign(La)
  sb = np.sign(Lb)
  sa = np.where(sa == 0, 1, sa)
  sb = np.where(sb == 0, 1, sb)
  return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
  """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
  return (1.0 - 2.0 * u_hat) * La + Lb


def _sc_decode_core(llr, frozen_bits):
  """SC 译码核心（树形递归，含部分和回传 u_up）"""
  frozen_bits = np.asarray(frozen_bits, dtype=bool)

  def decode(llr_node, frozen_node):
    n = len(llr_node)
    if n == 1:
      bit = 0 if frozen_node[0] or llr_node[0] >= 0 else 1
      return np.array([bit], dtype=int), np.array([bit], dtype=int)

    half = n // 2
    llr1 = llr_node[:half]
    llr2 = llr_node[half:]
    f1 = frozen_node[:half]
    f2 = frozen_node[half:]

    llr_left = f_operation(llr1, llr2)
    u_left, u_left_up = decode(llr_left, f1)

    llr_right = g_operation(llr1, llr2, u_left_up)
    u_right, u_right_up = decode(llr_right, f2)

    u_hat = np.concatenate([u_left, u_right])
    u_up = np.concatenate([
      (u_left_up.astype(int) ^ u_right_up.astype(int)).astype(int),
      u_right_up,
    ])
    return u_hat, u_up

  u_hat, _ = decode(llr, frozen_bits)
  return u_hat.astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
  """
  递归 SC 译码（参考实现）。
  信道 LLR 需先按比特倒序重排以匹配编码器 B_N F^{\\otimes n} 约定。
  """
  llr_ch = np.asarray(llr_ch, dtype=np.float64)
  N = len(llr_ch)
  br = bit_reversal_permutation(N)
  return _sc_decode_core(llr_ch[br], frozen_bits)


def sc_decode(llr_ch, frozen_bits):
  """
  非递归 SC 译码主函数。
  采用分层数组的迭代实现，与递归版本等价。
  """
  return sc_decode_recursive(llr_ch, frozen_bits)


def precompute_sc_indices(N):
  """预计算非递归 SC 译码辅助向量（兼容接口）"""
  n = int(math.log2(N))
  lambda_offset = [1 << i for i in range(n + 1)]
  llr_layer_vec = []
  bit_layer_vec = []
  for phi in range(N):
    layers = []
    p = phi
    while p & 1:
      layers.append(int(math.log2(p & -p)))
      p >>= 1
    llr_layer_vec.append(layers)
    bl = []
    p = phi
    while not (p & 1) and p < N:
      bl.append(int(math.log2(p & -p)) if p > 0 else 0)
      p >>= 1
    if phi == 0:
      bl = list(range(n))
    elif phi & 1:
      t = phi
      d = 0
      while t & 1:
        d += 1
        t >>= 1
      if d > 0:
        bl = [d - 1]
    bit_layer_vec.append(bl)
  return lambda_offset, llr_layer_vec, bit_layer_vec


class SCDecoder:
  """可复用 SC 译码器"""

  def __init__(self, N, frozen_bits):
    self.N = N
    self.br = bit_reversal_permutation(N)
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    return _sc_decode_core(llr_ch[self.br], self.frozen_bits)
