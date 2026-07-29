"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（sign(0)=+1）"""
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    if np.isscalar(u_hat) or (hasattr(u_hat, 'shape') and u_hat.shape == ()):
        b = int(u_hat)
        return La + Lb if b == 0 else La - Lb
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _apply_bit_reversal(llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br]


def _update_llrs(l, L, B, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                )


def _update_bits(l, B, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def precompute_sc_indices(N):
  """
  预计算非递归 SC / SCL 译码所需的辅助向量。
  使用 active-llr/bit-level 算法，避免 phi=0 时 pp%2==0 的死循环。
  """
  n = int(math.log2(N))
  lambda_offset = [1 << layer for layer in range(n + 1)]

  llr_layer_vec = []
  bit_layer_vec = []

  for phi in range(N):
      l = _bit_reversed_index(phi, n)
      llr_start = n - _active_llr_level(l, n)
      llr_layers = list(range(llr_start, n))
      if not llr_layers:
          llr_layers = [n - 1]
      llr_layer_vec.append(llr_layers)

      if l < N // 2:
          bit_layers = []
      else:
          bit_start = n - _active_bit_level(l, n)
          bit_layers = list(range(n, bit_start, -1))
      bit_layer_vec.append(bit_layers)

  return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，与非递归 sc_decode 等价）"""
    llr_ch = _apply_bit_reversal(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    def decode_phi(phi):
        if phi >= N:
            return
        l = _bit_reversed_index(phi, n)
        _update_llrs(l, L, B, n, N)
        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]
        _update_bits(l, B, n, N)
        decode_phi(phi + 1)

    decode_phi(0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（信道 LLR 自然顺序输入，内部做比特倒序）"""
    llr_ch = _apply_bit_reversal(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        _update_llrs(l, L, B, n, N)

        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        _update_bits(l, B, n, N)

    return u_hat


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(8.0, K / N)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u1, u2), "recursive vs non-recursive mismatch"
        assert np.array_equal(u1, u), "SC decode error"
    print("decoder_sc self-test passed")
