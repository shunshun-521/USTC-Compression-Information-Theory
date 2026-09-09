"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted SCD）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, bit_reversed_index


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


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


def _frozen_set_from_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return set(np.where(frozen_bits)[0])
    return set(np.where(frozen_bits.astype(int) > 0)[0])


def _sc_core(llr, frozen_set, n):
    """Permuted SCD 核心（参考 mcba1n / Vangala 2014）。"""
    N = len(llr)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            branch = block // 2
            for j in range(l, N, block):
                if j % block < branch:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch, s])
                else:
                    top_bit = B[j - branch, s + 1]
                    L[j, s + 1] = g_operation(L[j - branch, s], L[j, s], top_bit)

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block = 1 << s
            branch = block // 2
            for j in range(l, -1, -block):
                if j % block >= branch:
                    B[j - branch, s - 1] = int(B[j, s]) ^ int(B[j - branch, s])
                    B[j, s - 1] = B[j, s]

    for l in [bit_reversed_index(i, n) for i in range(N)]:
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    llr = np.asarray(llr_ch, dtype=np.float64)[br]
    return _sc_core(llr, _frozen_set_from_mask(frozen_bits), n)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口兼容）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        p = phi
        layer = 0
        while p % 2 == 1:
            llr_layers.append(layer)
            p //= 2
            layer += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        p = phi
        layer = 0
        while p % 2 == 0:
            bit_layers.append(layer)
            p //= 2
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 先按比特倒序重排，再执行 Permuted SCD。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    llr = np.asarray(llr_ch, dtype=np.float64)[br]
    return _sc_core(llr, _frozen_set_from_mask(frozen_bits), n)
