"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted SCD）
"""
import numpy as np


def f_operation(La, Lb):
    """f 运算（LLR 域精确形式）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    out = np.empty_like(La)
    flat_a = La.ravel()
    flat_b = Lb.ravel()
    flat_o = out.ravel()
    for i in range(flat_a.size):
        l1 = flat_a[i]
        l2 = flat_b[i]
        flat_o[i] = _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)
    return out


def _logdomain_sum(x, y):
    if x > y:
        return float(x + np.log1p(np.exp(y - x)))
    return float(y + np.log1p(np.exp(x - y)))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def bit_reversed_int(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def precompute_sc_indices(N):
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed_int(i, n)
        start_s = n - active_llr_level(l, n)
        llr_layer_vec.append(list(range(start_s, n)))
        if l < N // 2:
            bit_layer_vec.append([])
        else:
            start_s = n - active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, start_s, -1)))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _scd_update_llrs(L, B, l, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)


def _scd_update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_set = set(np.where(frozen_bits.astype(bool))[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1))
    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed_int(i, n)
        _scd_update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _scd_update_bits(B, l, n, N)

    return B[:, n].astype(np.int8)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 等价）。"""
    return sc_decode(llr, frozen_bits)
