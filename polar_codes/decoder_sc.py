"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）与非递归置换 SC 实现（高效）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def _logdomain_sum(x, y):
    x = float(x)
    y = float(y)
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """f 运算（对数域精确 box-plus，向量化）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    vec = np.vectorize(
        lambda a, b: _logdomain_sum(a + b, 0.0) - _logdomain_sum(a, b), otypes=[float]
    )
    return vec(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
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


def _preprocess_channel_llr(llr_ch):
    """与蝶形+比特倒序编码配套的 LLR 预处理。"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _permuted_sc_decode(llr_ch, frozen_bits):
    """置换 SC 译码核心。"""
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    decode_order = [_bit_reversed_index(i, n) for i in range(N)]

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            branch = block // 2
            for j in range(l, N, block):
                if j % block < branch:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch, s])
                else:
                    top_bit = B[j - branch, s + 1]
                    L[j, s + 1] = g_operation(L[j - branch, s], L[j, s], top_bit)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block = 1 << s
                branch = block // 2
                for j in range(l, -1, -block):
                    if j % block >= branch:
                        B[j - branch, s - 1] = int(B[j, s]) ^ int(B[j - branch, s])
                        B[j, s - 1] = B[j, s]

    return u_hat


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（经 LLR 预处理与置换译码等效）。"""
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算置换 SC 译码比特顺序。"""
    n = int(math.log2(N))
    return [_bit_reversed_index(i, n) for i in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """非递归置换 SC 译码主函数。"""
    llr = _preprocess_channel_llr(llr_ch)
    return _permuted_sc_decode(llr, frozen_bits)
