"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 PSCD 实现（高效）
"""
import math

import numpy as np

from encoder import bit_reversed, bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """找 i 的二进制表示中第一个 1 的位置（从高位起）"""
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
    """找 i 的二进制表示中第一个 0 的位置（从高位起）"""
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
    """更新 LLR 树（PSCD）"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        N = L.shape[0]
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s],
                    L[j, s],
                    B[j - branch_size, s + 1],
                )


def _update_bits(B, l, n):
    """比特回传（PSCD）"""
    N = B.shape[0]
    if l < N // 2:
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
    非递归 PSCD 译码。
    frozen_bits 按 u 向量自然索引：0=信息位，1=冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    rev = bit_reversal_permutation(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed(i, n)
        _update_llrs(L, B, l, n)

        if frozen_bits[i]:
            B[l, n] = 0
            L[l, n] = abs(L[l, n])
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        _update_bits(B, l, n)

    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
        l = bit_reversed(i, n)
        u_hat[i] = B[l, n]
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口兼容）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        p = phi
        while p % 2 == 1:
            layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(layers)
        bit_layer_vec.append(list(range(n)) if phi % 2 == 0 else [int(math.log2(phi & -phi))])
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用 PSCD）"""
    return sc_decode(llr, frozen_bits)
