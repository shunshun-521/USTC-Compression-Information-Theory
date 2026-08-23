"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversed


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr_scalar(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def f_operation(La, Lb):
    """精确 log-domain box-plus（f 运算）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0:
        return float(_upper_llr_scalar(float(La), float(Lb)))
    return np.vectorize(_upper_llr_scalar)(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = Lb + La (u=0) 或 Lb - La (u=1)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.int32)
    return np.where(u_hat == 0, Lb + La, Lb - La).astype(np.float64)


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


def _frozen_indices(frozen_bits):
    return set(np.where(np.asarray(frozen_bits, dtype=bool))[0])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return
        half = n // 2
        decode_node(f_operation(llr_node[:half], llr_node[half:]), bit_offset)
        u_left = u_hat[bit_offset:bit_offset + half]
        decode_node(g_operation(llr_node[:half], llr_node[half:], u_left), bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    lambda_offset = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
        lambda_offset.append(l)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（Permuted SCD）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_set = _frozen_indices(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch

    for phi in range(N):
        l = bit_reversed(phi, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_scalar(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    if top_bit == 0:
                        L[j, s + 1] = L[j, s] + L[j - branch_size, s]
                    else:
                        L[j, s + 1] = L[j, s] - L[j - branch_size, s]

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(np.int32)


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        u_src = np.zeros(N, dtype=int)
        u_src[info_idx] = rng.integers(0, 2, K)
        y = bpsk_modulate(polar_encode(u_src)) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_rec = sc_decode(llr, frozen)
        if not np.array_equal(u_src[info_idx], u_rec[info_idx]):
            errors += 1
    print(f"SC test: {errors} errors in 100 frames at Eb/N0=10dB")
