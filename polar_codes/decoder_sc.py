"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（使用部分和比特）"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """LLR 更新起始层"""
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
    """比特回传起始层"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_reversed(x, n):
    r = 0
    for i in range(n):
        if x & (1 << i):
            r |= 1 << (n - 1 - i)
    return r


def _prepare_channel_llrs(llr_ch, N):
    """
    编码器输出含比特倒序时，将信道 LLR 映射到译码树顺序。
    x[i] = v[br(i)] => LLR(v[j]) = llr_ch[br(j)]
    """
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，使用部分和）"""
    N = len(llr)
    n = int(math.log2(N))
    llr = _prepare_channel_llrs(llr, N)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block = 2 ** (s + 1)
            branch = block // 2
            for j in range(l, N, block):
                if j % block < branch:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch, s], L[j, s], B[j - branch, s + 1]
                    )

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block = 2 ** s
            branch = block // 2
            for j in range(l, -1, -block):
                if j % block >= branch:
                    B[j - branch, s - 1] = (B[j, s] + B[j - branch, s]) % 2
                    B[j, s - 1] = B[j, s]

    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
        l = _bit_reversed(i, n)
        update_llrs(l)
        u_hat[l] = 0 if frozen_bits[l] or L[l, n] >= 0 else 1
        B[l, n] = u_hat[l]
        update_bits(l)

    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [2 ** (layer - 1) if layer > 0 else 0 for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        layers_llr = list(range(n - _active_llr_level(l, n), n))
        layers_bit = list(range(n, n - _active_bit_level(l, n), -1)) if l >= N // 2 else []
        llr_layer_vec.append(layers_llr)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（高效实现，与递归版本等价）。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    llr = _prepare_channel_llrs(llr_ch, N)
    frozen_bits = np.asarray(frozen_bits, dtype=int)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = _bit_reversed(i, n)

        for s in range(n - _active_llr_level(l, n), n):
            block = 2 ** (s + 1)
            branch = block // 2
            for j in range(l, N, block):
                if j % block < branch:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch, s], L[j, s], B[j - branch, s + 1]
                    )

        u_hat[l] = 0 if frozen_bits[l] or L[l, n] >= 0 else 1
        B[l, n] = u_hat[l]

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block = 2 ** s
                branch = block // 2
                for j in range(l, -1, -block):
                    if j % block >= branch:
                        B[j - branch, s - 1] = (B[j, s] + B[j - branch, s]) % 2
                        B[j, s - 1] = B[j, s]

    return u_hat
