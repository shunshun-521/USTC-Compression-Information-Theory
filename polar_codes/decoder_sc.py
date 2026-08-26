"""
极化码 SC（串行抵消）译码器
置换 SC（Permuted SCD）实现，与蝶形编码器配套
"""
import math

import numpy as np

from encoder import bit_reversed


def f_operation(La, Lb):
    """box-plus 运算（LLR 域精确 f，小值时用 min-sum 近似）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    large = np.maximum(np.abs(La), np.abs(Lb)) > 30
    result = np.empty_like(La)
    if np.any(large):
        result[large] = (
            np.sign(La[large])
            * np.sign(Lb[large])
            * np.minimum(np.abs(La[large]), np.abs(Lb[large]))
        )
    if np.any(~large):
        ta = np.tanh(La[~large] / 2.0)
        tb = np.tanh(Lb[~large] / 2.0)
        prod = np.clip(ta * tb, -0.999999, 0.999999)
        result[~large] = 2.0 * np.arctanh(prod)
    return result


def g_operation(La, Lb, u_hat):
    """g 运算（lower LLR）：La=top, Lb=bottom。"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """二进制表示中第一个 1 的位置（从高位计）。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    """二进制表示中第一个 0 的位置（从高位计）。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归置换 SC 译码。
    frozen_bits: True/1 表示冻结位（强制为 0）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    decode_order = [bit_reversed(i, n) for i in range(N)]

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用置换 SC 作为参考）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：返回置换 SC 的辅助信息。"""
    n = int(math.log2(N))
    lambda_offset = [1 << l for l in range(n + 1)]
    llr_layer_vec = [
        list(range(n - _active_llr_level(bit_reversed(phi, n), n), n))
        for phi in range(N)
    ]
    bit_layer_vec = [
        list(range(n, n - _active_bit_level(bit_reversed(phi, n), n), -1))
        for phi in range(N)
    ]
    return lambda_offset, llr_layer_vec, bit_layer_vec
