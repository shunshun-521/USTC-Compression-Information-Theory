"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：La 为上半分支，Lb 为下半分支"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    rev = 0
    for _ in range(n):
        rev = (rev << 1) | (i & 1)
        i >>= 1
    return rev


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
    if (mask & i) > 0:
      count += 1
      mask >>= 1
    else:
      break
  return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset, length):
        if length == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return

        half = length // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset, half)

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half, half)

    decode_node(llr, 0, N)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（基于 Permuted SC 算法）。
    编码器输出含比特倒序置换，此处将信道 LLR 还原为蝶形顺序。
    """
    from encoder import bit_reversal_permutation

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    br = bit_reversal_permutation(N)
    llr_internal = llr_ch[br]

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_internal

    frozen_set = set(np.where(frozen_bits)[0])

    for phi in range(N):
        l = _bit_reversed(phi, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        top_bit,
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（兼容接口）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1)) if l >= N // 2 else []
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
