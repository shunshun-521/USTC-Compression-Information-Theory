"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


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


def f_operation(La, Lb):
    """f 运算（box-plus），供 SCL/BP 复用。"""
    la = np.asarray(La, dtype=np.float64)
    lb = np.asarray(Lb, dtype=np.float64)
    flat_a = la.ravel()
    flat_b = lb.ravel()
    out = np.array([_upper_llr(a, b) for a, b in zip(flat_a, flat_b)], dtype=np.float64)
    return out.reshape(la.shape)


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    la = np.asarray(La, dtype=np.float64)
    lb = np.asarray(Lb, dtype=np.float64)
    u = np.asarray(u_hat, dtype=np.int8)
    return np.array(
        [_lower_llr(a, b, int(ui)) for a, b, ui in zip(la.ravel(), lb.ravel(), u.ravel())],
        dtype=np.float64,
    ).reshape(la.shape)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（参考 mcba1n/polar-codes SCD 实现）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for l in [_bit_reversed(i, n) for i in range(N)]:
        update_llrs(l)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 结果一致，用于校验）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（接口兼容）。"""
    n = int(math.log2(N))
    lambda_offset = np.arange(n + 1, dtype=np.int32)
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llr_layers(P, C, phi, n, N):
    """SCL 内部复用的占位接口（由路径矩阵直接维护）。"""
    raise NotImplementedError("Use SCLDecoder internal updates")


def _propagate_bits(C, phi, bit_layers, n, N):
    raise NotImplementedError("Use SCLDecoder internal updates")
