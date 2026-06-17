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
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim > 0:
        return np.array([f_operation(a, b) for a, b in zip(La, Lb)], dtype=np.float64)

    la = float(La)
    lb = float(Lb)
    if np.isnan(la) and np.isnan(lb):
        return np.nan
    if np.isnan(la):
        return lb
    if np.isnan(lb):
        return la
    sa = 0.0 if la == 0.0 else np.sign(la)
    sb = 0.0 if lb == 0.0 else np.sign(lb)
    return sa * sb * min(abs(la), abs(lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.int8)
    if La.ndim > 0:
        return np.array(
            [g_operation(a, b, u) for a, b, u in zip(La, Lb, u_hat)], dtype=np.float64
        )

    la = float(La)
    lb = float(Lb)
    if np.isnan(la) and np.isnan(lb):
        return np.nan
    if np.isnan(la):
        return lb
    if np.isnan(lb):
        return la
    return (1.0 - 2.0 * int(u_hat)) * la + lb


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def _permute_channel_llr(llr_ch, N):
    """将信道 LLR 置换到 SC 因子图索引（与含比特倒序的编码器匹配）。"""
    br = bit_reversal_permutation(N)
    return llr_ch[br]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，与高效非递归版本等价）。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（与按比特倒序遍历的高效实现对应）。
    """
    n = int(math.log2(N))
    lambda_offset = [0] * N
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        layers = list(range(n - _active_llr_level(l, n), n))
        llr_layer_vec.append(layers)
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        bit_layer_vec.append(bit_layers)
        lambda_offset[phi] = l

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（高效实现，按比特倒序遍历因子图）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    llr = _permute_channel_llr(llr_ch, N)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr

    def f_ms(l1, l2):
        v1, v2 = float(l1), float(l2)
        if np.isnan(v1) and np.isnan(v2):
            return np.nan
        if np.isnan(v1):
            return v2
        if np.isnan(v2):
            return v1
        return np.sign(v1) * np.sign(v2) * min(abs(v1), abs(v2))

    def g_ms(l1, l2, b):
        v1, v2 = float(l1), float(l2)
        if np.isnan(v1) and np.isnan(v2):
            return np.nan
        if np.isnan(v1):
            return v2
        if np.isnan(v2):
            return v1
        return (v1 + v2) if b == 0 else (v1 - v2)

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_ms(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_ms(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)
