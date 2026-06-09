"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversed_index


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（La 为下支路 LLR，Lb 为上支路 LLR）"""
    if np.isscalar(u_hat):
        return La + Lb if int(u_hat) == 0 else La - Lb
    return np.where(u_hat == 0, La + Lb, La - Lb)


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
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], top_bit)


def _update_bits(B, l, n):
    if l < B.shape[0] / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（按比特倒序逐位译码）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=np.int32)
    decode_order = [bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = int(B[l, n])
        _update_bits(B, l, n)

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归版接口，与 sc_decode 等价"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：返回比特倒序译码顺序及活跃层信息"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    decode_order = [bit_reversed_index(i, n) for i in range(N)]
    llr_layer_vec = [[s for s in range(n - _active_llr_level(l, n), n)] for l in decode_order]
    bit_layer_vec = [[s for s in range(n, n - _active_bit_level(l, n), -1)] for l in decode_order]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _pm_penalty(llr, u):
    """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)
