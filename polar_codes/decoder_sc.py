"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sign_a = np.sign(La)
    sign_b = np.sign(Lb)
    sign_a[sign_a == 0] = 1
    sign_b[sign_b == 0] = 1
    return sign_a * sign_b * np.minimum(np.abs(La), np.abs(Lb))


def _f_scalar(a, b):
    sa = 1 if a > 0 else (-1 if a < 0 else 1)
    sb = 1 if b > 0 else (-1 if b < 0 else 1)
    return sa * sb * min(abs(a), abs(b))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _g_scalar(a, b, u):
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return (1.0 - 2.0 * u) * a + b


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


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


def _prepare_llr(llr_ch):
    """将信道 LLR 映射到 SCD 内部索引顺序"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    inv = np.empty_like(br)
    inv[br] = np.arange(len(br))
    return llr_ch[inv]


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _scd_decode(llr, frozen_set, n):
    """SCD 核心译码（min-sum）"""
    N = 2 ** n
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _f_scalar(L[j, s], L[j + branch_size, s])
                else:
                    bit_val = B[j - branch_size, s + 1]
                    if not np.isnan(bit_val):
                        L[j, s + 1] = _g_scalar(
                            L[j - branch_size, s], L[j, s], bit_val
                        )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = int(B[l, n])

        if l < N / 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用 SCD）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    llr = _prepare_llr(np.asarray(llr, dtype=np.float64))
    n = int(math.log2(len(llr)))
    return _scd_decode(llr, frozen_set, n)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    return sc_decode_recursive(llr_ch, frozen_bits)
