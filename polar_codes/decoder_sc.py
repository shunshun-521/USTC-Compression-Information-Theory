"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """找到 i 的二进制表示中第一个 1 的位置（从高位起）"""
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
    """找到 i 的二进制表示中第一个 0 的位置（从高位起）"""
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
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    N = len(llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    br = bit_reversal_permutation(N)
    y = llr[br].astype(np.float64)
    u_list = []

    def sc_rec(llr_node, depth, frozen_seg):
        if depth == 0:
            u_list.append(0 if frozen_seg[0] else (0 if llr_node[0] >= 0 else 1))
            return
        half = 1 << (depth - 1)
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        sc_rec(llr_left, depth - 1, frozen_seg[:half])
        u_left = np.array(u_list[-half:], dtype=int)
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        sc_rec(llr_right, depth - 1, frozen_seg[half:])

    sc_rec(y, n, frozen_bits)
    return np.array(u_list, dtype=int)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回 active_llr_level 与 active_bit_level 查找表。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = bit_reversal_permutation(N)[phi]
        start_llr = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start_llr, n)))
        start_bit = n - _active_bit_level(l, n) + 1
        bit_layer_vec.append(list(range(n, start_bit - 1, -1)) if l >= N // 2 else [])

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（permute-order，自然信道 LLR 顺序）"""
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr_ch.astype(np.float64)

    br = bit_reversal_permutation(N)
    u_hat = np.zeros(N, dtype=np.int32)

    for phi in range(N):
        l = br[phi]
        _update_llrs(L, B, l, n)

        if frozen_bits[phi]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        _update_bits(B, l, n, N)
        u_hat[phi] = B[l, n]

    return u_hat


if __name__ == "__main__":
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    print(f"SC test errors: {errors}/100")
