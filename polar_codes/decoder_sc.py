"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，置换 SC）
"""
import math

import numpy as np

from encoder import bit_reversed_index

LLR_CLIP = 30.0


def clip_llr(llr):
    return np.clip(np.asarray(llr, dtype=np.float64), -LLR_CLIP, LLR_CLIP)


def logdomain_sum(x, y):
    """对数域加法"""
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_exact(La, Lb):
    """精确 f 运算（对数域，用于递归参考实现）"""
    if np.isscalar(La) and np.isscalar(Lb):
        return logdomain_sum(La + Lb, 0.0) - logdomain_sum(La, Lb)
    return np.vectorize(lambda a, b: logdomain_sum(a + b, 0.0) - logdomain_sum(a, b))(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算（置换 SC 约定：La=下分支 LLR，Lb=上分支 LLR）。
    u=0: La + Lb；u=1: La - Lb
    """
    u_hat = np.asarray(u_hat)
    if np.isscalar(u_hat) or u_hat.ndim == 0:
        return La + Lb if int(u_hat) == 0 else La - Lb
    return np.where(u_hat == 0, La + Lb, La - Lb)


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


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，置换顺序）"""
    llr_ch = clip_llr(llr_ch)
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
    N = len(llr_ch)
    n = int(math.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            half = block // 2
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = f_operation_exact(L[j, s], L[j + half, s])
                else:
                    L[j, s + 1] = g_operation(L[j, s], L[j - half, s], B[j - half, s + 1])

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block = 1 << s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    B[j - half, s - 1] = int(B[j, s]) ^ int(B[j - half, s])
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = bit_reversed_index(i, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = int(B[l, n])
        update_bits(l)

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（置换 SC）。
    frozen_bits: 1 表示冻结位，0 表示信息位
    """
    llr_ch = clip_llr(llr_ch)
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = bit_reversed_index(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            half = block // 2
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
                else:
                    L[j, s + 1] = g_operation(L[j, s], L[j - half, s], B[j - half, s + 1])

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = int(B[l, n])

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block = 1 << s
                half = block // 2
                for j in range(l, -1, -block):
                    if j % block >= half:
                        B[j - half, s - 1] = int(B[j, s]) ^ int(B[j - half, s])
                        B[j, s - 1] = B[j, s]

    return u_hat


def precompute_sc_indices(N):
    """保留接口：置换 SC 使用 active level 函数，无需额外预计算"""
    n = int(math.log2(N))
    return list(range(N)), [[] for _ in range(N)], [[] for _ in range(N)]
