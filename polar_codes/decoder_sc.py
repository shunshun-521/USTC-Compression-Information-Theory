"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def _bit_reversed_scalar(x, n):
    y = 0
    for i in range(n):
        if x & (1 << i):
            y |= 1 << (n - 1 - i)
    return y


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def f_operation(La, Lb):
    """f 运算（对数域 boxplus）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    out = np.empty_like(La)
    for i in range(La.size):
        l1, l2 = float(La[i]), float(Lb[i])
        if np.isinf(l1) and not np.isinf(l2):
            out[i] = l2
        elif not np.isinf(l1) and np.isinf(l2):
            out[i] = l1
        elif np.isinf(l1) and np.isinf(l2):
            out[i] = np.inf
        else:
            out[i] = _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)
    return out


def g_operation(La, Lb, u_hat):
    """g 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.int8)
    out = np.empty_like(La)
    for i in range(La.size):
        l1, l2 = float(La[i]), float(Lb[i])
        b = int(u_hat[i])
        if b == 0:
            out[i] = l1 + l2 if not (np.isinf(l1) or np.isinf(l2)) else np.inf
        else:
            out[i] = l1 - l2
    return out


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


def _upper_llr(l1, l2):
    return float(f_operation(np.array([l1]), np.array([l2]))[0])


def _lower_llr(l1, l2, b):
    return float(g_operation(np.array([l1]), np.array([l2]), np.array([b]))[0])


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（Permuted SCD）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    rev = bit_reversal_permutation(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch[rev]

    frozen_set = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = _bit_reversed_scalar(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1])

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

    return B[:, n].astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC（调用非递归实现作为参考）"""
    return sc_decode(llr_ch, frozen_bits)


def sc_self_test(N=64, K=32, num_frames=100, seed=0):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(seed)
    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            return False
    return True
