"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（置换 SC，高效实现）
"""
import math
import numpy as np
from encoder import bit_reversed_index


def f_operation(La, Lb):
    """min-sum 近似 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


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


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（标准半分 f/g），用于对照验证。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode(L, frozen):
        N = len(L)
        if N == 1:
            return np.array([0 if frozen[0] else (0 if L[0] >= 0 else 1)], dtype=int)
        h = N // 2
        L_left = f_operation(L[:h], L[h:])
        u_left = decode(L_left, frozen[:h])
        L_right = g_operation(L[:h], L[h:], u_left)
        u_right = decode(L_right, frozen[h:])
        return np.concatenate([u_left, u_right])

    return decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助结构（置换译码顺序）。"""
    n = int(math.log2(N))
    lambda_offset = [2 ** (i - 1) for i in range(n + 1)]
    decode_order = [bit_reversed_index(i, n) for i in range(N)]
    llr_layer_vec = [
        list(range(n - _active_llr_level(bit_reversed_index(phi, n), n), n))
        for phi in range(N)
    ]
    bit_layer_vec = [
        list(range(n, n - _active_bit_level(bit_reversed_index(phi, n), n), -1))
        for phi in range(N)
    ]
    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


def sc_decode(llr_ch, frozen_bits):
    """
    非递归置换 SC 译码（与蝶形编码配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (
                        B[j, s] ^ B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    for phi in range(N):
        l = bit_reversed_index(phi, n)
        update_llrs(l)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]
        update_bits(l)

    return u_hat


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 1e-3)
    frozen = np.zeros(4, dtype=bool)
    uh = sc_decode(llr, frozen)
    assert np.array_equal(uh, u), f"N=4 SC fail: {uh}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if np.array_equal(u_hat[info_idx], u[info_idx]):
            ok += 1
    print(f"SC high-SNR test: {ok}/100 frames correct")
    assert ok >= 95
