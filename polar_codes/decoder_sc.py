"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Permuted SCD，高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def _bit_reversed(i, n):
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


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


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（box-plus 的快速近似）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_exact(La, Lb):
    """精确 log-domain box-plus（f 运算）。"""
    if np.isscalar(La) and np.isscalar(Lb):
        if np.isinf(La) and not np.isinf(Lb):
            return Lb
        if not np.isinf(La) and np.isinf(Lb):
            return La
        if np.isinf(La) and np.isinf(Lb):
            return np.inf
        return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)
    return np.vectorize(f_operation_exact, otypes=[float])(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    等价于 u=0 时 La+Lb，u=1 时 La-Lb
    """
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def g_operation_exact(La, Lb, u_hat):
    """精确 log-domain g 运算。"""
    if u_hat == 0:
        if np.isinf(La) or np.isinf(Lb):
            return np.inf
        return La + Lb
    return La - Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（Permuted SCD 顺序）。
    frozen_bits: True/1 表示冻结位
    """
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = set(np.where(np.asarray(frozen_bits))[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation_exact(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation_exact(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = _bit_reversed(i, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（Permuted SCD）。
    """
    n = int(math.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = [_active_llr_level(l, n) for l in decode_order]
    bit_layer_vec = [_active_bit_level(l, n) for l in decode_order]
    return decode_order, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SCD 译码主函数。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(np.asarray(frozen_bits))[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    decode_order, _, _ = precompute_sc_indices(N)

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation_exact(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation_exact(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
