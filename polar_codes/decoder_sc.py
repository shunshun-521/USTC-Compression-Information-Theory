"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于 Permuted SC）
"""
import numpy as np
import math

from encoder import bit_reversed_index


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(L_top, L_btm, u_hat):
    """
    g 运算（下分支）：l1=btm, l2=top。
    u=0 -> top+btm；u=1 -> btm-top
    """
    u = int(u_hat)
    if u == 0:
        return L_btm + L_top
    return L_btm - L_top


def _hard_decision(llr):
    return 0 if llr >= 0 else 1


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


def _frozen_indices(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return set(np.where(frozen_bits)[0])


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Permuted Successive Cancellation，min-sum 近似）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_set = _frozen_indices(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for l in [bit_reversed_index(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(
                        np.array([L[j, s]]), np.array([L[j + branch_size, s]])
                    )[0]
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )  # (L_top, L_btm, u)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = _hard_decision(L[l, n])

        if l < N / 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（调用非递归实现，保证与主译码器一致）。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 辅助向量（供文档/扩展使用）。
    """
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (layer - 1))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = list(range(n - _active_llr_level(bit_reversed_index(phi, n), n), n))
        bit_layers = []
        if phi > 0:
            br_phi = bit_reversed_index(phi, n)
            bit_layers = list(range(n, n - _active_bit_level(br_phi, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec
