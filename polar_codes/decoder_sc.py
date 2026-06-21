"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from channel import channel_llr_to_decode


def f_operation(La, Lb):
  """min-sum 近似的 f 运算"""
  sa = np.sign(La)
  sb = np.sign(Lb)
  sa = np.where(sa == 0, 1, sa)
  sb = np.where(sb == 0, 1, sb)
  return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
  """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
  return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
  return int(f"{i:0{n}b}"[::-1], 2)


def _active_llr_level(i, n):
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
  mask = 1 << (n - 1)
  count = 1
  for _ in range(n):
    if mask & i:
      count += 1
      mask >>= 1
    else:
      break
  return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
  """递归 SC 译码（参考实现）"""
  llr = channel_llr_to_decode(np.asarray(llr, dtype=np.float64)).copy()
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  N = len(llr)
  u_hat = np.zeros(N, dtype=int)

  def decode_node(llr_node, bit_offset):
    n = len(llr_node)
    if n == 1:
      idx = bit_offset
      if frozen_bits[idx]:
        u_hat[idx] = 0
      else:
        u_hat[idx] = 0 if llr_node[0] >= 0 else 1
      return

    half = n // 2
    llr_left = f_operation(llr_node[:half], llr_node[half:])
    decode_node(llr_left, bit_offset)

    u_left = u_hat[bit_offset:bit_offset + half]
    llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
    decode_node(llr_right, bit_offset + half)

  decode_node(llr, 0)
  return u_hat


def precompute_sc_indices(N):
  """预计算非递归 SC 译码所需的辅助向量"""
  n = int(math.log2(N))
  lambda_offset = [1 << i for i in range(n + 1)]
  llr_layer_vec = []
  bit_layer_vec = []
  for phi in range(N):
    llr_start = n - _active_llr_level(phi, n)
    llr_layer_vec.append(list(range(llr_start, n)))
    bit_start = n - _active_bit_level(phi, n)
    bit_layer_vec.append(list(range(n, bit_start, -1)))
  return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs(L, B, l, n):
  for s in range(n - _active_llr_level(l, n), n):
    block_size = 1 << (s + 1)
    branch_size = block_size // 2
    for j in range(l, L.shape[0], block_size):
      if j % block_size < branch_size:
        L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
      else:
        L[j, s + 1] = g_operation(
          L[j - branch_size, s],
          L[j, s],
          B[j - branch_size, s + 1],
        )


def _update_bits(B, l, n):
  if l < B.shape[0] // 2:
    return
  for s in range(n, n - _active_bit_level(l, n), -1):
    block_size = 1 << s
    branch_size = block_size // 2
    for j in range(l, -1, -block_size):
      if j % block_size >= branch_size:
        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
        B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
  """
  非递归 SC 译码主函数。
  信道 LLR 经逆比特倒序置换后，按标准顺序更新、以倒序索引回传比特。
  """
  llr_ch = channel_llr_to_decode(np.asarray(llr_ch, dtype=np.float64))
  frozen_bits = np.asarray(frozen_bits, dtype=bool)
  N = len(llr_ch)
  n = int(math.log2(N))

  L = np.zeros((N, n + 1), dtype=np.float64)
  B = np.zeros((N, n + 1), dtype=np.int8)
  L[:, 0] = llr_ch

  for phi in range(N):
    l = _bit_reversed(phi, n)
    _update_llrs(L, B, l, n)
    if frozen_bits[l]:
      B[l, n] = 0
    else:
      B[l, n] = 0 if L[l, n] >= 0 else 1
    _update_bits(B, l, n)

  return B[:, n].astype(int)
