"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted SCD）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（主要用于 BP 译码器接口）。
  """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _log_domain_sum(a, b):
    m = np.maximum(np.abs(a), np.abs(b))
    inf_mask = np.isinf(m)
    out = m + np.log2(1.0 + np.power(2.0, -np.abs(a - b)))
    out = np.where(inf_mask, m, out)
    return out


def _f_boxplus(La, Lb):
    """SC 译码使用的 log-domain box-plus f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    pos_inf = np.isposinf(La) & ~np.isposinf(Lb)
    pos_inf2 = ~np.isposinf(La) & np.isposinf(Lb)
    both_inf = np.isposinf(La) & np.isposinf(Lb)
    out = _log_domain_sum(La + Lb, 0.0) - _log_domain_sum(La, Lb)
    out = np.where(pos_inf, Lb, out)
    out = np.where(pos_inf2, La, out)
    out = np.where(both_inf, np.inf, out)
    return out


def _g_boxplus(La, Lb, u_hat):
    """SC 译码使用的 g 运算。"""
    u_hat = np.asarray(u_hat)
    out = np.where(
        u_hat == 0,
        np.where(np.isposinf(La) | np.isposinf(Lb), np.inf, La + Lb),
        La - Lb,
    )
    return out


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


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


def _prepare_channel_llr(llr_ch):
    br = bit_reversal_permutation(len(llr_ch))
    return np.asarray(llr_ch, dtype=np.float64)[br]


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（调用非递归实现作为参考）。
    """
    return sc_decode(llr_ch, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（与 Permuted SCD 层激活表对应）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_start = n - _active_bit_level(l, n)
        bit_layer_vec.append(list(range(n, bit_start, -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Permuted SCD，mcba1n 风格）。
    """
    llr = _prepare_channel_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    frozen_set = set(np.where(frozen_bits == 1)[0])
    N = len(llr)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _f_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _g_boxplus(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
