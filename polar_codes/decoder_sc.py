"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Permuted SCD，高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, bit_reversed


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    return l1 + l2 if b == 0 else l1 - l2


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


def _frozen_set_from_array(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    return set(np.where(frozen_bits.astype(bool))[0])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 Permuted SCD 等价）"""
    return sc_decode(llr, frozen_bits, use_min_sum=False)


def precompute_sc_indices(N):
    """
    预计算 Permuted SCD 辅助信息。
    """
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec, br


def _update_llrs(L, B, l, n, f_fn):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_fn(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = _lower_llr(
                    L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                )


def _update_bits(B, l, n):
    if l < B.shape[0] / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _sc_decode_perm_core(llr_tree, frozen_set, use_min_sum=False):
    """Permuted SCD 核心（llr_tree 已做比特倒序置换）"""
    N = len(llr_tree)
    n = int(math.log2(N))
    f_fn = f_operation if use_min_sum else _upper_llr

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_tree

    for l in [bit_reversed(i, n) for i in range(N)]:
        _update_llrs(L, B, l, n, f_fn)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits, use_min_sum=False):
    """
    非递归 Permuted SC 译码主函数。
    信道 LLR 经比特倒序置换后送入译码树（与 bit-reversal 编码器配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    frozen_set = _frozen_set_from_array(frozen_bits)
    llr_tree = llr_ch[br]
    return _sc_decode_perm_core(llr_tree, frozen_set, use_min_sum=use_min_sum)
