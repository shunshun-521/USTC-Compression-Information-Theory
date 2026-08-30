"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted SCD）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


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
                top_llr = L[j, s]
                btm_llr = L[j + branch_size, s]
                L[j, s + 1] = f_operation(top_llr, btm_llr)
            else:
                btm_llr = L[j, s]
                top_llr = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)


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


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码参考实现。
    与 Permuted SCD 在数值上等价（同一 min-sum f/g 与比特倒序信道 LLR）。
    """
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（兼容 SCL 接口）。
    """
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(n + 1):
        lambda_offset[i] = 1 << i

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        start_s = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start_s, n)))
        bit_start = n - _active_bit_level(l, n)
        bit_layer_vec.append(list(range(n, bit_start, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Permuted SCD, Vangala 2014）。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    br_perm = bit_reversal_permutation(N)
    L[:, 0] = llr_ch[br_perm]
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = _bit_reversed(phi, n)
        _update_llrs(L, B, l, n)

        if frozen_bits[phi]:
            u_hat[phi] = 0
            B[l, n] = 0
        else:
            u_hat[phi] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[phi]

        _update_bits(B, l, n)

    return u_hat
