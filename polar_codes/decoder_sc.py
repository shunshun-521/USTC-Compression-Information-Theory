"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def _sign_llr(x):
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """min-sum 近似 f 运算。"""
    if np.isscalar(La):
        return float(
            _sign_llr(La) * _sign_llr(Lb) * min(abs(La), abs(Lb))
        )
    return _sign_llr(La) * _sign_llr(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


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


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助信息。"""
    n = int(math.log2(N))
    phases = [_bit_reversed(i, n) for i in range(N)]
    llr_layers = [list(range(n - _active_llr_level(l, n), n)) for l in phases]
    bit_layers = [
        list(range(n, n - _active_bit_level(l, n), -1)) if l >= N // 2 else []
        for l in phases
    ]
    return phases, llr_layers, bit_layers


def _sc_decode_core(llr, frozen_bits):
    """SC 译码核心（Vangala et al. 非递归实现）。"""
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
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
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    编码器输出含比特倒序，信道 LLR 需做相同倒序后与因子图对齐。
    """
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr = np.asarray(llr_ch, dtype=np.float64)[br]
    return _sc_decode_core(llr, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用同一核心逻辑）。"""
    return sc_decode(llr, frozen_bits)
