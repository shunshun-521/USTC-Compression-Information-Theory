"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _to_frozen_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return frozen_bits
    return frozen_bits.astype(bool)


def _prepare_channel_llrs(llr_ch, N):
    """编码端含比特倒序时，将信道 LLR 映射到译码树索引。"""
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现，算法与 sc_decode 一致）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_mask = _to_frozen_mask(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_mask)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = _prepare_channel_llrs(llr_ch, N)
    u_hat = np.zeros(N, dtype=int)

    def update_llrs_rec(l, s):
        if s >= n:
            return
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )
        update_llrs_rec(l, s + 1)

    def update_bits_rec(l, s, s_stop):
        if s <= s_stop:
            return
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]
        update_bits_rec(l, s - 1, s_stop)

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        update_llrs_rec(l, n - _active_llr_level(l, n))

        if l in frozen_set:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        if l >= N // 2:
            update_bits_rec(l, n, n - _active_bit_level(l, n))

    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（与层索引 s 对应）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        layers_llr = list(range(n - _active_llr_level(l, n), n))
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if l >= N // 2:
            layers_bit = list(range(n, n - _active_bit_level(l, n), -1))
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（高效实现）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_mask = _to_frozen_mask(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_mask)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = _prepare_channel_llrs(llr_ch, N)

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        _update_llrs(L, B, l, n)

        if l in frozen_set:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        _update_bits(B, l, n, N)

    return u_hat


if __name__ == "__main__":
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(10.0, K / N)
    errors_rec = errors_fast = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        if not np.array_equal(sc_decode_recursive(llr, frozen_bits)[info_idx], u[info_idx]):
            errors_rec += 1
        if not np.array_equal(sc_decode(llr, frozen_bits)[info_idx], u[info_idx]):
            errors_fast += 1
    print(f"SC recursive errors: {errors_rec}/100")
    print(f"SC non-recursive errors: {errors_fast}/100")
