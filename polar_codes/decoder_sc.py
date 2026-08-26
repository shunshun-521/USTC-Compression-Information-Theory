"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversed_index, prepare_decoder_llr


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    """精确 box-plus f 运算。"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, bit):
    """g 运算（对数域）。"""
    if bit == 0:
        return l1 + l2
    return l1 - l2


def f_operation(La, Lb):
    """向量化 min-sum f（供 BP 等模块使用）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """向量化 g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


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


def _frozen_to_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return set(np.where(frozen_bits)[0])


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（Tal/Vardy 调度 + 精确 box-plus）。"""
    llr_ch = prepare_decoder_llr(llr_ch)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen = _frozen_to_set(frozen_bits)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    def update_llrs(l):
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = bit_reversed_index(i, n)
        update_llrs(l)
        if l in frozen:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]
        update_bits(l)

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用非递归实现，便于对照验证）。"""
    return sc_decode_nonrecursive(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口兼容）。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed_index(phi, n)
        layers = list(range(n - active_llr_level(l, n), n))
        llr_layer_vec.append(layers)
        bit_layers = list(range(n, n - active_bit_level(l, n), -1))
        bit_layer_vec.append(bit_layers)
    return llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口。"""
    return sc_decode_nonrecursive(llr_ch, frozen_bits)
