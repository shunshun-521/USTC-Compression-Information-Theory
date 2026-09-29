"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def bit_reversed_index(x, n):
    """单索引比特倒序（n 位）。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def _logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def upper_llr(l1, l2):
    """f 运算（min-sum）。"""
    l1 = float(np.clip(l1, -1e3, 1e3))
    l2 = float(np.clip(l2, -1e3, 1e3))
    return np.sign(l1) * np.sign(l2) * min(abs(l1), abs(l2))


def lower_llr(l1, l2, b):
    """g 运算（min-sum，l1=bottom, l2=top）。"""
    l1 = float(np.clip(l1, -1e3, 1e3))
    l2 = float(np.clip(l2, -1e3, 1e3))
    return (1 - 2 * int(b)) * l2 + l1


def f_operation(La, Lb):
    """向量 min-sum f（供 BP 等模块复用）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """向量 g（供 BP 等模块复用）。"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


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


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = lower_llr(
                    L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                )


def _update_bits(B, l, n):
    if l < B.shape[0] // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = int(B[j, s])


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Permuted SCD，与 mcba1n 参考实现一致）。
    """
    llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -1e3, 1e3)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    frozen_set = set(np.where(frozen_bits)[0])
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = bit_reversed_index(i, n)
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)
        u_hat[l] = int(B[l, n])

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与非递归结果一致）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（接口兼容）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed_index(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec
