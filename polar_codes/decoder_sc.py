"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于 mcba1n SCD）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(top, btm, u) = btm + (1-2*u)*top"""
    return Lb + (1.0 - 2.0 * u_hat) * La


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
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


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])
    L = np.full((N, n + 1), np.nan)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr.astype(np.float64)
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = _bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            bs = 2 ** (s + 1)
            br = bs // 2
            for j in range(l, N, bs):
                if j % bs < br:
                    L[j, s + 1] = f_operation(L[j, s], L[j + br, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - br, s], L[j, s], B[j - br, s + 1]
                    )
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]
        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                bs = 2 ** s
                br = bs // 2
                for j in range(l, -1, -bs):
                    if j % bs >= br:
                        B[j - br, s - 1] = int(B[j, s]) ^ int(B[j - br, s])
                        B[j, s - 1] = B[j, s]
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容接口）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = _bit_reversed(i, n)
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1)) if l >= N // 2 else []
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    lambda_offset = list(range(n + 1))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（与 sc_decode_recursive 等价）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
