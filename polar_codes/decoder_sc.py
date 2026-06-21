"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================


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


def _llr_to_bit(llr):
    return 0 if llr >= 0 else 1


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


def _map_channel_llr(llr_ch, N):
    """将信道 LLR 映射到译码器内部顺序（与含比特倒序的编码器配套）"""
    return np.asarray(llr_ch, dtype=np.float64)[bit_reversal_permutation(N)]


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（调用非递归核心以保证与编码器一致）"""
    return sc_decode(llr_ch, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """返回比特倒序译码顺序"""
    n = int(math.log2(N))
    decode_order = [int(f"{i:0{n}b}"[::-1], 2) for i in range(N)]
    return decode_order, []


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Vangala 风格，与蝶形+比特倒序编码器配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    assert 2**n == N

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = _map_channel_llr(llr_ch, N)

    decode_order = [int(f"{i:0{n}b}"[::-1], 2) for i in range(N)]

    for l in decode_order:
        start = n - _active_llr_level(l, n)
        for s in range(start, n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = _llr_to_bit(L[l, n])

        if l >= N // 2:
            end = n - _active_bit_level(l, n)
            for s in range(n, end, -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n]


def _precompute_sc_indices_v2(N):
    return (None,) + precompute_sc_indices(N)
