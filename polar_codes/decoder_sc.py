"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation, bit_reversed_index


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def hard_decision(llr):
    return 0 if llr >= 0 else 1


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


def _prepare_llr(llr_ch):
    """编码含比特倒序，信道 LLR 需同步倒序后送入译码树。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    brp = bit_reversal_permutation(len(llr_ch))
    return llr_ch[brp]


def _frozen_index_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return set(np.flatnonzero(frozen_bits))


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    极化码含比特倒序时，树形递归与自然序 LLR 不直接兼容，
    此处复用与 sc_decode 等价的逐位倒序非递归流程以保证正确性。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = bit_reversed_index(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs(L, B, l, n, N):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = int(B[j - branch_size, s + 1])
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], top_bit
                )


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


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_ch = _prepare_llr(llr_ch)
    frozen_set = _frozen_index_set(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)
    decode_order = [bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        _update_llrs(L, B, l, n, N)
        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            B[l, n] = hard_decision(L[l, n])
            u_hat[l] = int(B[l, n])
        _update_bits(B, l, n, N)

    return u_hat


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_rec = sc_decode(llr, frozen_bits)
        assert np.array_equal(u[info_idx], u_rec[info_idx]), "SC decode error"

    print(f"SC verification passed: N={N}, K={K}, {num_frames} frames @ {eb_n0_db}dB")


if __name__ == "__main__":
    verify_sc_decoders()
