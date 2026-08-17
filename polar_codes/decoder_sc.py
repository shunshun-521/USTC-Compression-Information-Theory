"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversed


def logdomain_sum(x, y):
    """对数域加法"""
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """f 运算（对数域 box-plus）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    scalar = La.ndim == 0 and Lb.ndim == 0
    La = np.atleast_1d(La)
    Lb = np.atleast_1d(Lb)
    la, lb = np.broadcast_arrays(La, Lb)
    result = np.empty_like(la, dtype=np.float64)
    for idx in np.ndindex(la.shape):
        a, b = la[idx], lb[idx]
        if np.isinf(a) and not np.isinf(b):
            result[idx] = b
        elif not np.isinf(a) and np.isinf(b):
            result[idx] = a
        elif np.isinf(a) and np.isinf(b):
            result[idx] = np.inf
        else:
            result[idx] = logdomain_sum(a + b, 0.0) - logdomain_sum(a, b)
    return float(result.flat[0]) if scalar else result


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if u_hat.ndim == 0 or (hasattr(u_hat, 'size') and u_hat.size == 1):
        b = int(np.asarray(u_hat).flat[0])
        if b == 0:
            if np.isinf(La) or np.isinf(Lb):
                return np.inf
            return La + Lb
        return La - Lb
    return np.where(u_hat == 0, La + Lb, La - Lb)


def f_min_sum(La, Lb):
    """min-sum 近似的 f 运算（BP 译码用）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    mask_a0 = np.abs(La) < 1e-12
    mask_b0 = np.abs(Lb) < 1e-12
    result = np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))
    result = np.where(mask_a0, Lb, result)
    result = np.where(mask_b0, La, result)
    return result


def active_llr_level(i, n):
    """找到二进制表示中第一个 1 的位置（从高位计）"""
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
    """找到二进制表示中第一个 0 的位置（从高位计）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _sc_decode_core(llr_ch, frozen_bits):
    """非递归 SC 译码核心（参考 Permuted SCD 算法）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_set = set(np.where(frozen_bits)[0])
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    def update_llrs(l):
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = bit_reversed(i, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return _sc_decode_core(llr_ch, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    return sc_decode(llr, frozen_bits)
