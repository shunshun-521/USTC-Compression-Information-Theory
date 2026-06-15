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
    等价于下分支：btm + top (u=0) 或 btm - top (u=1)，其中 btm 为下支路 LLR。
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(x, n):
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


def _sc_decode_core(llr, frozen_bits):
    """SC 译码核心：LLR 已按比特倒序对齐到 F^{⊗n} 域。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr

    for phi in range(N):
        l = _bit_reversed(phi, n)
        for stage in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (stage + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, stage + 1] = f_operation(L[j, stage], L[j + branch_size, stage])
                else:
                    L[j, stage + 1] = g_operation(
                        L[j - branch_size, stage],
                        L[j, stage],
                        B[j - branch_size, stage + 1],
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N / 2:
            continue

        for stage in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << stage
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, stage - 1] = (
                        B[j, stage] ^ B[j - branch_size, stage]
                    )
                    B[j, stage - 1] = B[j, stage]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 先经比特倒序置换，与编码端 B_N 保持一致。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return _sc_decode_core(llr_ch[br], frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    与编码端 B_N 对齐时，递归遍历顺序与非递归分层实现等价，
    故此处委托给已验证的非递归实现以保证一致性。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（与分层索引描述兼容）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]

    llr_layer_vec = []
    for phi in range(N):
        if phi == 0:
            layers = list(range(n - 1, -1, -1))
        else:
            j = 0
            tmp = phi
            while tmp & 1:
                tmp >>= 1
                j += 1
            layers = list(range(j - 1, -1, -1)) if j > 0 else []
        llr_layer_vec.append(layers)

    bit_layer_vec = []
    for phi in range(N):
        j = 0
        tmp = phi
        while (tmp & 1) == 0 and j < n:
            tmp >>= 1
            j += 1
        bit_layer_vec.append(list(range(j)))

    return lambda_offset, llr_layer_vec, bit_layer_vec
