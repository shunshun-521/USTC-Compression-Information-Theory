"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

_INF = np.float64("inf")


def f_operation(La, Lb):
  """min-sum 近似的 f 运算"""
  return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
  """g 运算"""
  return (1 - 2 * u_hat) * La + Lb


def _b_check(layer, idx):
  return (idx // (1 << layer)) & 1


def _s_updater(layer, idx, s):
  if _b_check(layer - 1, idx):
    s[layer, idx] = s[layer - 1, idx]
  else:
    if s[layer - 1, idx] < 0:
      _s_updater(layer - 1, idx, s)
    sib = idx + (1 << (layer - 1))
    if s[layer - 1, sib] < 0:
      _s_updater(layer - 1, sib, s)
    s[layer, idx] = (s[layer - 1, idx] ^ s[layer - 1, sib]) & 1


def _compute_llr(layer, idx, llrs, s):
  if llrs[layer, idx] > -_INF / 2:
    return llrs[layer, idx]
  if _b_check(layer, idx) == 0:
    left = _compute_llr(layer + 1, idx, llrs, s)
    right = _compute_llr(layer + 1, idx + (1 << layer), llrs, s)
    llrs[layer, idx] = f_operation(left, right)
  else:
    if layer > 0:
      _s_updater(layer, idx - (1 << layer), s)
    left_idx = idx - (1 << layer)
    left = _compute_llr(layer + 1, left_idx, llrs, s)
    right = _compute_llr(layer + 1, idx, llrs, s)
    llrs[layer, idx] = g_operation(left, right, s[layer, left_idx])
  return llrs[layer, idx]


def sc_decode_recursive(llr, frozen_bits):
  """递归 SC 译码（参考实现）"""
  N = len(llr)
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  u_hat = np.zeros(N, dtype=int)

  def decode_node(llr_node, bit_offset):
    n = len(llr_node)
    if n == 1:
      idx = bit_offset
      u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
      return
    half = n // 2
    llr_left = f_operation(llr_node[:half], llr_node[half:])
    decode_node(llr_left, bit_offset)
    llr_right = g_operation(
      llr_node[:half], llr_node[half:], u_hat[bit_offset : bit_offset + half]
    )
    decode_node(llr_right, bit_offset + half)

  decode_node(np.asarray(llr, dtype=np.float64), 0)
  return u_hat


def precompute_sc_indices(N):
  """预计算非递归 SC 译码所需的辅助向量"""
  n = int(math.log2(N))
  lambda_offset = np.arange(N, dtype=int)
  llr_layer_vec = []
  bit_layer_vec = []

  for phi in range(N):
    if phi == 0:
      llr_layers = list(range(n))
    else:
      llr_layers = []
      tmp = phi
      layer = 0
      while tmp % 2 == 0:
        llr_layers.append(layer)
        tmp //= 2
        layer += 1
    llr_layer_vec.append(llr_layers)

    bit_layers = []
    tmp = phi
    layer = 0
    while (tmp & 1) == 1:
      bit_layers.append(layer)
      tmp >>= 1
      layer += 1
    bit_layer_vec.append(bit_layers)

  return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
  """
  非递归 SC 译码（惰性 LLR 计算，O(N log N)）。
  """
  N = len(llr_ch)
  n = int(math.log2(N))
  frozen_bits = np.asarray(frozen_bits, dtype=bool)

  llrs = np.full((n + 1, N), -_INF, dtype=np.float64)
  llrs[n, :] = np.asarray(llr_ch, dtype=np.float64)
  s = np.full((n + 1, N), -1, dtype=np.int32)

  u_hat = np.zeros(N, dtype=np.int32)
  for phi in range(N):
    if frozen_bits[phi]:
      u_hat[phi] = 0
      s[0, phi] = 0
      llrs[0, phi] = _INF
    else:
      llr_val = _compute_llr(0, phi, llrs, s)
      u_hat[phi] = 1 if llr_val < 0 else 0
      s[0, phi] = u_hat[phi]
  return u_hat


if __name__ == "__main__":
  from construction import ga_construction
  from encoder import polar_encode
  from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0

  rng = np.random.default_rng(0)
  sigma = eb_n0_to_sigma(10.0, K / N)
  errors = 0
  for _ in range(100):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    y = bpsk_modulate(x) + rng.normal(0, sigma, N)
    llr = compute_llr(y, sigma)
    u_rec = sc_decode(llr, frozen_bits)
    if not np.array_equal(u[info_idx], u_rec[info_idx]):
      errors += 1
  print(f"SC test: {errors}/100 errors at Eb/N0=10dB")
