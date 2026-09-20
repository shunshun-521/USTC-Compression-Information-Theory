"""
极化码 SC（串行抵消）译码器
非递归实现（mcba1n / Vangala 风格），与 encoder.polar_encode 配套
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """Box-plus（log-domain）"""
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def g_operation(Lb, La, u_hat):
    """g 运算，Lb 为下层 LLR，La 为上层 LLR"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, Lb + La, Lb - La)


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


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j, s],
                    L[j - branch_size, s],
                    B[j - branch_size, s + 1],
                )


def _update_bits(B, l, n, N):
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
    非递归 SC 译码。
    llr_ch: 信道 LLR（LLR = 2y/σ²，正号倾向比特 0）
    frozen_bits: 长度 N，1 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)
    decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    for l in decode_order:
        _update_llrs(L, B, l, n, N)
        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit
        _update_bits(B, l, n, N)

    return u_hat


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归版本（调用非递归实现）"""
    return sc_decode(llr_ch, frozen_bits)
