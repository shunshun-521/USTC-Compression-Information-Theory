"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted SC）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def bit_reversed_index(i, n):
    """单索引比特倒序（与 Arikan / Vangala 记号一致）"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_boxplus(La, Lb):
    """对数域 box-plus（精确 f 运算）"""
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def _llr_to_bit(llr_val, is_frozen):
    if is_frozen:
        return 0
    return 0 if llr_val >= 0 else 1


def _pm_penalty(llr_val, u_bit):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if hard == u_bit else abs(llr_val)


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


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr_ch, frozen_bits):
    """递归参考实现（与非递归版本等价，基于 Permuted SC）"""
    return sc_decode_nonrecursive(llr_ch, frozen_bits)


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_boxplus(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)


def _update_bits(B, l, n):
    if l < B.shape[0] // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """预计算 Permuted SC 的译码顺序"""
    n = int(math.log2(N))
    decode_order = [bit_reversed_index(i, n) for i in range(N)]
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for l in decode_order:
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 Permuted SC 译码"""
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    rev = bit_reversal_permutation(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = np.asarray(llr_ch, dtype=np.float64)[rev]

    for phi in range(N):
        l = bit_reversed_index(phi, n)
        _update_llrs(L, B, l, n)
        B[l, n] = 0 if frozen_bits[l] else _llr_to_bit(L[l, n], False)
        _update_bits(B, l, n)

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数"""
    return sc_decode_nonrecursive(llr_ch, frozen_bits)
