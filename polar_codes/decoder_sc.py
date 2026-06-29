"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from decoder_utils import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    f_minsum,
    g_llr,
    hard_decision,
)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return f_minsum(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return g_llr(La, Lb, u_hat)


def _prepare_channel_llr(llr_ch, N):
    """将信道 LLR 调整为译码树所需顺序。"""
    rev = np.array([bit_reversed(i, int(np.log2(N))) for i in range(N)])
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = _prepare_channel_llr(llr, len(llr))
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = hard_decision(llr_node[0])
            return
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        for i in range(half):
            decode_node(llr_left[i:i + 1], bit_offset + i)
        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i:i + 1], bit_offset + half + i)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = list(range(n - active_llr_level(phi, n), n))
        bit_layers = list(range(n, n - active_bit_level(phi, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（Permuted SCD）。"""
    llr_ch = _prepare_channel_llr(llr_ch, len(llr_ch))
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed(i, n)
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    B_top = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], B_top)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = hard_decision(L[l, n])

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 下有 {errors} 帧错误"
    print("SC 译码校验通过")
