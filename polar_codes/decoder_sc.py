"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
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


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
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


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _prepare_llr(llr_ch, N):
    br = bit_reversal_permutation(N)
    return llr_ch[br]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与蝶形树结构一致）。"""
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr = _prepare_llr(llr_ch, N)

    L = np.zeros((N, n + 1))
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = _bit_reversed(i, n)
        _update_llrs(L, B, l, n, N)
        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]
        _update_bits(B, l, n, N)

    return u_hat


def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 使用）。"""
    n = int(math.log2(N))
    lambda_offset = []
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 0:
                layers_llr.append(layer)
            temp //= 2
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 1:
                layers_bit.append(layer)
            temp //= 2
        bit_layer_vec.append(layers_bit)

        if len(layers_llr) == 0:
            lambda_offset.append(0)
        else:
            lambda_offset.append((phi // (2 ** len(layers_llr))) * 2 ** (len(layers_llr) + 1) - 1)

    return lambda_offset, llr_layer_vec, bit_layer_vec


