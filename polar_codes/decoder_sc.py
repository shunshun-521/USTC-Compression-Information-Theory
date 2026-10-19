"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，置换 SC）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _frozen_to_bool(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb.astype(bool)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    frozen_bits: True 表示冻结位。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen = _frozen_to_bool(frozen_bits)
    N = llr.size
    decisions = [0] * N

    def _sc(llrs, indices):
        if len(indices) == 1:
            i = indices[0]
            if frozen[i]:
                decisions[i] = 0
            else:
                decisions[i] = 0 if llrs[0] >= 0 else 1
            return [decisions[i]]

        half = len(indices) // 2
        left_llr = [
            f_operation(llrs[i], llrs[i + half]) for i in range(half)
        ]
        enc_left = _sc(left_llr, indices[:half])
        right_llr = [
            g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
        ]
        enc_right = _sc(right_llr, indices[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    _sc(llr.tolist(), list(range(N)))
    return np.array(decisions, dtype=int)


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def _active_llr_level(i, n):
    """与 py-polar-codes `active_llr_level` 一致。"""
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


def _upper_llr(top, btm):
    return f_operation(top, btm)


def _lower_llr(btm, top, top_bit):
    return g_operation(top, btm, top_bit)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助结构（供报告/扩展使用）。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(
            list(range(n, n - _active_bit_level(l, n), -1))
        )
    return list(range(N)), llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """
    非递归置换 SC 译码（Vangala / py-polar-codes SCD 结构）。
    frozen_bits: 1=冻结，0=信息位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bool = _frozen_to_bool(frozen_bits)
    N = llr_ch.size
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    frozen_set = set(np.where(frozen_bool)[0])

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for phi_nat in range(N):
        l = _bit_reversed(phi_nat, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)


sc_decode = sc_decode_nonrecursive
