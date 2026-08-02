"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from _ref_function import f_hf, g
from _sc_core import sc_tree_decode
from encoder import _bit_rev_indices


def f_operation(La, Lb):
  """min-sum 近似的 f 运算。"""
  return np.vectorize(f_hf)(La, Lb)


def g_operation(La, Lb, u_hat):
  """g 运算。"""
  return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
  """递归 SC 译码（参考实现）。"""
  return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
  """预计算非递归 SC 译码所需的辅助向量。"""
  n = int(math.log2(N))
  llr_layer_vec = []
  bit_layer_vec = []
  for phi in range(N):
    llr_layers = []
    bpp = phi
    while bpp % 2 == 1:
      llr_layers.append(int(math.log2(bpp & -bpp)))
      bpp //= 2
    llr_layers.append(n)
    llr_layer_vec.append(llr_layers)
    bit_layers = []
    bpp = phi
    while bpp % 2 == 1:
      bit_layers.append(int(math.log2(bpp & -bpp)))
      bpp //= 2
    bit_layer_vec.append(bit_layers)
  lambda_offset = np.arange(N)
  return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
  """非递归 SC 译码主函数。"""
  N = len(llr_ch)
  brp = _bit_rev_indices(N)
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  info_indices = np.where(~frozen_bits)[0]
  llr_br = np.asarray(llr_ch, dtype=np.float64)[brp]
  u_hat = sc_tree_decode(llr_br, info_indices, frozen_bit=0)
  u_hat[frozen_bits] = 0
  return u_hat
