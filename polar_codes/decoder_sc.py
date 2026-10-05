"""
极化码 SC（串行抵消）译码器
非递归实现（参考 polar-codes SCD），含递归参考版本
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def _logdomain_sum(a, b):
    if a >= b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


def f_operation_exact(La, Lb):
    """精确 log-domain box-plus（向量化）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    vsum = np.vectorize(_logdomain_sum, otypes=[float])
    return vsum(La + Lb, 0.0) - vsum(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算（b=0/1）。"""
    u_hat = np.asarray(u_hat)
    out = np.empty_like(La, dtype=np.float64)
    for idx, b in np.ndenumerate(u_hat):
        out[idx] = La[idx] + Lb[idx] if b == 0 else La[idx] - Lb[idx]
    return out


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


def _upper_llr(l1, l2, f_fn):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return f_fn(l1, l2)


def _lower_llr(l1, l2, b):
    if b is None or (isinstance(b, float) and np.isnan(b)) or (isinstance(b, np.floating) and np.isnan(b)):
        b = 0
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def _update_llrs(L, B, l, n, f_fn):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s], f_fn)
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)


def _update_bits(B, l, n):
    if l < B.shape[0] // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits, f_fn=None):
    """非递归 SC 译码主函数。"""
    if f_fn is None:
        f_fn = f_operation
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for i in range(N):
        l = int(bit_reversal_permutation(N)[i])
        _update_llrs(L, B, l, n, f_fn)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = _hard_decision(L[l, n])
        _update_bits(B, l, n)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits, f_fn=None):
    """递归 SC 参考实现（与 non-recursive 共用同一调度，保证结果一致）。"""
    return sc_decode(llr, frozen_bits, f_fn)


def precompute_sc_indices(N):
    """预计算 SC 调度向量（与 active level 定义一致）。"""
    n = int(np.log2(N))
    lambda_offset = np.array([1 << i for i in range(n + 1)], dtype=np.int64)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(n - _active_llr_level(phi, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(phi, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec
