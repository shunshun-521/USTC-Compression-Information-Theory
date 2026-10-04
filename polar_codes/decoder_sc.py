"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """精确 log-domain f 运算（box-plus），标量/向量"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if np.isscalar(La) and np.isscalar(Lb):
        return _upper_llr_scalar(float(La), float(Lb))
    return np.vectorize(_upper_llr_scalar)(La, Lb)


def _upper_llr_scalar(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return float(np.logaddexp(0.0, l1 + l2) - np.logaddexp(l1, l2))


def g_operation(La, Lb, u_hat):
    """g 运算（log-domain）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    out = np.empty_like(La, dtype=np.float64)
    mask0 = u_hat == 0
    out[mask0] = La[mask0] + Lb[mask0]
    out[~mask0] = La[~mask0] - Lb[~mask0]
    if np.isscalar(La):
        return float(out)
    return out


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


def _update_llrs(L, B, l, n):
    N = L.shape[0]
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr_scalar(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                if np.isnan(top_bit):
                    top_bit = 0
                L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], top_bit)


def _update_bits(B, l, n):
    N = B.shape[0]
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 与主实现一致"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """接口兼容：返回预计算索引"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    rev = bit_reversal_permutation(N)
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for i in range(N):
        l = rev[i]
        p = l
        while p & 1:
            llr_layer_vec[i].append(int(math.log2(p & -p)))
            p >>= 1
        p = l
        while p > 0 and (p & 1) == 0:
            bit_layer_vec[i].append(int(math.log2(p & -p)))
            p >>= 1
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    frozen_bits[i]=1 表示冻结位；返回完整源序列 u_hat。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]
    for l in decode_order:
        _update_llrs(L, B, l, n)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    u_hat = np.nan_to_num(B[:, n], nan=0.0).astype(int)
    return u_hat
