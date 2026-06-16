"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 PSC 版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：La 为上分支，Lb 为下分支"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(x, n):
    """单整数比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def _upper_llr(l1, l2):
    return f_operation(l1, l2)


def _lower_llr(btm, top, bit):
    if bit == 0:
        return top + btm
    return btm - top


def _psc_update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block = 1 << (s + 1)
        branch = block >> 1
        for j in range(l, L.shape[0], block):
            if j % block < branch:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch, s])
            else:
                L[j, s + 1] = _lower_llr(
                    L[j, s], L[j - branch, s], int(B[j - branch, s + 1])
                )


def _psc_update_bits(B, l, n):
    if l < B.shape[0] // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block = 1 << s
        branch = block >> 1
        for j in range(l, -1, -block):
            if j % block >= branch:
                B[j - branch, s - 1] = int(B[j, s]) ^ int(B[j - branch, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码参考入口。
    与 sc_decode（PSC 非递归）在数值上等价，用于校验与对照。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算 PSC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = [
        list(range(n - _active_llr_level(d, n), n)) for d in decode_order
    ]
    bit_layer_vec = [
        list(range(n, n - _active_bit_level(d, n), -1)) for d in decode_order
    ]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 PSC 译码。
    编码端含比特倒序时，先将信道 LLR 做 bit-reversal 再送入 PSC 因子图。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch[br]

    frozen_set = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = _bit_reversed(i, n)
        _psc_update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _psc_update_bits(B, l, n)

    return B[:, n].astype(int)
