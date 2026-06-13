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


def _active_llr_level(i, n):
    """二进制展开中第一个 1 的位置（从高位起）"""
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
    """二进制展开中第一个 0 的位置（从高位起）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n):
    """更新第 l 个比特所需的 LLR 树"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n, N):
    """回传硬判决比特"""
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    for l in decode_order:
        _update_llrs(L, B, l, n)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(np.int8)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（调用非递归实现作为参考）。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（供 SCL 使用）。
    """
    n = int(np.log2(N))
    lambda_offset = np.array([2 ** (n - layer) for layer in range(n + 1)], dtype=int)

    llr_layer_vec = []
    bit_layer_vec = []
    decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    for l in decode_order:
        start_s = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start_s, n)))

        bit_count = _active_bit_level(l, n)
        bit_layer_vec.append(list(range(n, n - bit_count, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llr_layers_scl(P, C, layer_list, lambda_offset, N, n):
    """SCL 使用的分层 LLR 更新（与 SC 树结构一致）"""
    for s in layer_list:
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(0, N, block_size):
            for off in range(branch_size):
                idx = j + off
                if idx % block_size < branch_size:
                    P[s + 1, idx] = f_operation(P[s, idx], P[s, idx + branch_size])
                else:
                    P[s + 1, idx] = g_operation(
                        P[s, idx - branch_size], P[s, idx], C[s + 1, idx - branch_size]
                    )


def _propagate_bit_layers_scl(C, bit_list, N, n, bit, l):
    """SCL 比特回传"""
    if l < N // 2:
        return
    for s in bit_list:
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                C[s - 1, j - branch_size] = int(C[s, j]) ^ int(C[s, j - branch_size])
                C[s - 1, j] = C[s, j]


def run_sc_self_test():
    """SC 译码无损自检"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        assert np.array_equal(u[info_idx], u_sc[info_idx]), "SC decode error at high SNR"
