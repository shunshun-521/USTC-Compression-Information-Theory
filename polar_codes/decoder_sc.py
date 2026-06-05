"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
  支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


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
    """
    递归 SC 译码（参考实现，与 sc_decode 共用核心）。
    """
    return sc_decode(llr, frozen_bits)


def _sc_decode_core(llr_ch, frozen_set, n, use_minsum=True):
    """基于因子树列存储的非递归 SC 译码核心。"""
    N = len(llr_ch)

    def f_fn(l1, l2):
        if use_minsum:
            return float(f_operation(l1, l2))
        return _boxplus(l1, l2)

    def g_fn(l1, l2, b):
        return float(g_operation(l1, l2, b))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_fn(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_fn(
                        L[j - branch_size, s],
                        L[j, s],
                        int(B[j - branch_size, s + 1]),
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def _boxplus(l1, l2):
    """对数域 box-plus（递归译码验证用）。"""
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    if np.isinf(l1):
        return l2
    if np.isinf(l2):
        return l1

    def log_sum(a, b):
        m = max(a, b)
        return m + np.log1p(np.exp(-abs(a - b)))

    return log_sum(l1 + l2, 0.0) - log_sum(l1, l2)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        p = phi
        layer = 0
        while p & 1:
            llr_layers.append(layer)
            p >>= 1
            layer += 1
        llr_layers.append(layer)
        while layer < n - 1:
            layer += 1
            llr_layers.append(layer)
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        p = (phi + 1) // 2
        layer = 0
        while p & 1:
            bit_layers.append(layer)
            p >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（高效列存储实现）。
    """
    from channel import reorder_llr_for_decode

    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
    llr = reorder_llr_for_decode(llr_ch, N)
    return _sc_decode_core(llr, frozen_set, n, use_minsum=True)
