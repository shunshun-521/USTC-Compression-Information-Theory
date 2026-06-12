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
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """比特索引 i 在 LLR 更新中需要处理的层数。"""
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
    """比特索引 i 在比特回传中需要处理的层数。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
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
                B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                B[j, s - 1] = B[j, s]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（比特倒序处理顺序）。
    """
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    lambda_offset = [1 << layer for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for step in range(N):
        l = br[step]
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（比特倒序信道顺序，与编码器配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for step in range(N):
        l = br[step]
        _update_llrs(L, B, l, n, N)

        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        _update_bits(B, l, n, N)

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，与 sc_decode 结果一致）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)

    llr_br = llr[br]
    frozen_br = frozen_bits[br]

    u_br = np.zeros(N, dtype=int)

    def decode_node(llr_node, depth, bit_offset):
        size = len(llr_node)
        if size == 1:
            idx = bit_offset
            if frozen_br[idx]:
                u_br[idx] = 0
            else:
                u_br[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = size // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left = np.zeros(half, dtype=int)
        for i in range(half):
            decode_node(llr_left[i:i + 1], depth - 1, bit_offset + i)
            u_left[i] = u_br[bit_offset + i]

        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i:i + 1], depth - 1, bit_offset + half + i)

    decode_node(llr_br, n, 0)

    u_hat = np.zeros(N, dtype=int)
    u_hat[br] = u_br
    return u_hat


if __name__ == "__main__":
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    u_test = np.array([1, 0, 1, 1])
    x_test = polar_encode(u_test)
    assert np.array_equal(x_test, [1, 1, 0, 1]), f"编码器错误: {x_test}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    print(f"SC noiseless test: {errors} errors in 100 frames")
    assert errors == 0
