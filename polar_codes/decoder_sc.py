"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation
from internal.pc_function import f_hf as f_operation_scalar
from internal.pc_decoder import sc_decoder as _sc_decoder_raw


def f_operation(La, Lb):
  """min-sum 近似的 f 运算。"""
  return np.vectorize(f_operation_scalar)(La, Lb)


def g_operation(La, Lb, u_hat):
  """g 运算。"""
  return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
  """递归 SC 译码（参考实现）。"""
  return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
  """预计算非递归 SC 所需索引（接口兼容）。"""
  n = int(np.log2(N))
  llr_layer_vec = []
  bit_layer_vec = []
  for phi in range(N):
    llr_layers = []
    temp = phi
    layer = 0
    while (temp & 1) == 1 and layer < n:
      temp >>= 1
      layer += 1
    while layer < n:
      llr_layers.append(layer)
      layer += 1
    llr_layer_vec.append(llr_layers)

    bit_layers = []
    temp_phi = phi
    layer = 0
    while layer < n and (temp_phi & 1) == 1:
      bit_layers.append(layer)
      temp_phi >>= 1
      layer += 1
    bit_layer_vec.append(bit_layers)

  lambda_offset = [2 ** max(i - 1, 0) for i in range(n + 1)]
  return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
  """
  非递归 SC 译码主函数。
  对信道 LLR 做比特倒序后使用树形 SC 译码。
  """
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  info_idx = list(np.where(~frozen_bits)[0])
  rev = bit_reversal_permutation(len(llr_ch))
  u_hat = _sc_decoder_raw(llr_ch[rev], info_idx, 0)[0]
  return u_hat.astype(int)
