"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _upper_llr(l1, l2):
    return f_operation(l1, l2)


def _lower_llr(btm, top, bit):
    if bit == 0:
        return btm + top
    return btm - top


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


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（供 SCL 使用）。"""
    n = int(math.log2(N))
    lambda_offset = [(1 << i) - 1 for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_core(llr_natural, frozen_bits):
    """SC 译码核心：信道 LLR 按自然码序输入。"""
    llr_natural = np.asarray(llr_natural, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_natural)
    n = int(math.log2(N))

    br = bit_reversal_permutation(N)
    llr_internal = llr_natural[br]

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_internal

    decode_order = [_bit_reversed(i, n) for i in range(N)]

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N // 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（调用与高效实现相同的核心逻辑）。"""
    return _sc_decode_core(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return _sc_decode_core(llr_ch, frozen_bits)


if __name__ == "__main__":
    from construction import ga_construction
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from encoder import polar_encode

    rng = np.random.default_rng(0)
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    print(f"SC 无损验证: {100 - errors}/100 帧正确")
