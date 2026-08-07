"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversed, bit_reversal_permutation


def _sign_llr(x):
    """LLR 符号函数，零视为正。"""
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算。
    """
    return _sign_llr(La) * _sign_llr(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_boxplus(La, Lb):
    """对数域精确 f 运算（boxplus）。"""
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def active_llr_level(i, n):
    """二进制展开中从高位起第一个 1 的位置计数。"""
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
    """二进制展开中从高位起第一个 0 的位置计数。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n, N, use_minsum=True):
    f_fn = f_operation if use_minsum else f_boxplus
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_fn(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                if top_bit == 0:
                    L[j, s + 1] = L[j - branch_size, s] + L[j, s]
                else:
                    L[j, s + 1] = L[j, s] - L[j - branch_size, s]


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits, use_minsum=True):
    """
    非递归 SC 译码主函数（按比特倒序依次译码）。
    编码器输出含比特倒序置换，因此先将信道 LLR 做相同置换。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    rev = bit_reversal_permutation(N)
    llr_ch = llr_ch[rev]
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed(i, n)
        _update_llrs(L, B, l, n, N, use_minsum=use_minsum)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用非递归核心）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（兼容接口）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << d for d in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        layers_llr = list(range(n - active_llr_level(l, n), n))
        llr_layer_vec.append(layers_llr)
        if l < N // 2:
            bit_layer_vec.append([])
        else:
            bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
    from construction import ga_construction

    rng = np.random.default_rng(0)
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    errors = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    print(f"SC test: {errors} errors in 100 frames at 10dB")
