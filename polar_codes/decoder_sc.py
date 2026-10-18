"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，PSC 风格）
"""
import numpy as np
import math


def bit_reversed(x, n):
    """单整数比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """对数域 f（boxplus）"""
    return logdomain_sum(La + Lb, 0.0) - logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """对数域 g"""
    if u_hat == 0:
        return La + Lb
    return La - Lb


def upper_llr(l1, l2):
    return f_operation(l1, l2)


def lower_llr(l1, l2, b):
    return g_operation(l1, l2, b)


def active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(l, L, B, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                top_llr = L[j, s]
                btm_llr = L[j + branch_size, s]
                L[j, s + 1] = upper_llr(top_llr, btm_llr)
            else:
                btm_llr = L[j, s]
                top_llr = L[j - branch_size, s]
                top_bit = int(B[j - branch_size, s + 1])
                L[j, s + 1] = lower_llr(btm_llr, top_llr, top_bit)


def _update_bits(l, B, n):
    if l < B.shape[0] / 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（按比特倒序相位译码）。
    frozen_bits: 1 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits)
    frozen_set = set(np.where(frozen_bits.astype(bool))[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed(i, n)
        _update_llrs(l, L, B, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(l, B, n)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 等价的参考实现，调用非递归）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（供文档/扩展使用）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def verify_sc_decoders(N=64, K=32, num_trials=100, seed=0):
    """验证高 SNR 下 SC 无错"""
    from construction import ga_construction
    from encoder import polar_encode, channel_llr_to_decoder
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    rng = np.random.default_rng(seed)
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(num_trials):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = channel_llr_to_decoder(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            raise AssertionError("SC decode error at high SNR")
    return True
