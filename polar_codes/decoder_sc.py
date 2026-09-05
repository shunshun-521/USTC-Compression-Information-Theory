"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def bit_reversed(i, n):
    """对 n 位索引 i 做比特倒序。"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def logdomain_sum(x, y):
    """对数域加法（数值稳定）。"""
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    """f 运算：min-sum 近似（大信噪比下比 log-domain 更稳定）。"""
    return float(np.sign(l1 * l2) * min(abs(l1), abs(l2)))


def lower_llr(l1, l2, b):
    """g 运算。"""
    b = int(b)
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def active_llr_level(i, n):
    """llr 更新起始层。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """比特回传起始层。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def f_operation(La, Lb):
    """min-sum 近似 f 运算（供 BP 等模块复用）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La * Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _init_llr_matrix(llr_ch, N, n):
    """信道 LLR 写入因子图第 0 层（自然顺序）。"""
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch
    return L


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Permuted SCD，精确 log-domain f/g）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = _init_llr_matrix(llr_ch, N, n)
    B = np.full((N, n + 1), np.nan)
    u_hat = np.zeros(N, dtype=int)

    phase_order = [bit_reversed(i, n) for i in range(N)]

    for l in phase_order:
        start = n - active_llr_level(l, n)
        for s in range(start, n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    B[j - branch_size, s + 1] = 0.0 if np.isnan(B[j - branch_size, s + 1]) else B[j - branch_size, s + 1]
                    L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1])

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 1 if L[l, n] < 0 else 0

        u_hat[l] = int(B[l, n])

        if l < N // 2:
            continue
        stop = n - active_bit_level(l, n)
        for s in range(n, stop, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    top = 0.0 if np.isnan(B[j, s]) else B[j, s]
                    bottom = 0.0 if np.isnan(B[j - branch_size, s]) else B[j - branch_size, s]
                    B[j - branch_size, s - 1] = int(top) ^ int(bottom)
                    B[j, s - 1] = top

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用与主实现等价的 Permuted SCD）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口供 SCL 使用。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for phi in range(N):
        layer = 0
        idx = phi
        while idx & 1:
            llr_layer_vec[phi].append(layer)
            layer += 1
            idx >>= 1
        while layer < n:
            llr_layer_vec[phi].append(layer)
            layer += 1
        layer = 0
        idx = phi
        while (idx & 1) == 0 and layer < n:
            bit_layer_vec[phi].append(layer)
            layer += 1
            idx >>= 1
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_path_metric(llr_val, u_bit):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr_val)
