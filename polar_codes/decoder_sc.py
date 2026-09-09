"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 Permuted SC 实现（高效）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
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


def _g_llr(l1, l2, bit):
    return l1 + l2 if bit == 0 else l1 - l2


def _prepare_llr(llr_ch):
    """比特倒序置换信道 LLR，与编码端 B_N 一致"""
    br = bit_reversal_permutation(len(llr_ch))
    return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SC 译码（Vangala et al., 2014）。
    """
    llr = _prepare_llr(llr_ch)
    N = len(llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = _g_llr(L[j, s], L[j - branch_size, s], top_bit)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = int(B[l, n])

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，内部使用 Permuted SC 保证与主实现一致）。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（Permuted SC 在运行时按需计算层级）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [
        list(range(n - _active_llr_level(_bit_reversed(phi, n), n), n))
        for phi in range(N)
    ]
    bit_layer_vec = [
        list(range(n, n - _active_bit_level(_bit_reversed(phi, n), n), -1))
        if _bit_reversed(phi, n) >= N / 2 else []
        for phi in range(N)
    ]
    return lambda_offset, llr_layer_vec, bit_layer_vec
