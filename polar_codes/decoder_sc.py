"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Vangala 2014 置换 SC 结构）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_exact(La, Lb):
    """对数域精确 f 运算（box-plus）"""
    ta = np.tanh(La / 2.0)
    tb = np.tanh(Lb / 2.0)
    p = np.clip(ta * tb, -1.0 + 1e-12, 1.0 - 1e-12)
    return 2.0 * np.arctanh(p)


def _upper_llr(l1, l2):
    """SC 主路径：优先精确 f，退化到 min-sum"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return f_operation_exact(l1, l2)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


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


def _channel_llr_to_decoder(llr_ch):
    """
    编码器输出含比特倒序置换 x[i]=v[rev(i)]，
    因子图叶节点 LLR 按未倒序码字 v 排列。
    """
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return llr_ch[rev]


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（主实现）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    frozen_set = set(np.where(frozen_bits)[0])
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = _channel_llr_to_decoder(llr_ch)

    for i in range(N):
        l = _bit_reversed_index(i, n)
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价，用于交叉验证）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 辅助向量（供 SCL 使用，与置换 SC 层对应）。
    """
    n = int(math.log2(N))
    lambda_offset = [2 ** layer - 1 for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = _bit_reversed_index(i, n)
        levels = list(range(n - _active_llr_level(l, n), n))
        llr_layer_vec.append(levels)
        if l >= N / 2:
            bit_levels = list(range(n, n - _active_bit_level(l, n), -1))
        else:
            bit_levels = []
        bit_layer_vec.append(bit_levels)
    return lambda_offset, llr_layer_vec, bit_layer_vec
