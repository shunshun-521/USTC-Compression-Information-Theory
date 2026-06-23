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
    等价于 u_hat=0 时 La+Lb，u_hat=1 时 La-Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """自高位起第一个 1 之前连续 0 的个数（含首个 1 位）"""
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
    """自高位起第一个 0 之前连续 1 的个数（含首个 0 位）"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_reversed(i, n):
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def _channel_llr_layout(llr_ch, N):
    """将传输顺序 LLR 映射到译码树自然节点顺序"""
    br = bit_reversal_permutation(N)
    br_inv = np.empty(N, dtype=int)
    br_inv[br] = np.arange(N)
    return np.asarray(llr_ch, dtype=np.float64)[br_inv]


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（与主译码器共享同一 LLR 布局约定）。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr = _channel_llr_layout(llr_ch, N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr

    for phi in range(N):
        l = _bit_reversed(phi, n)
        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            half = block >> 1
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
                else:
                    top_bit = B[j - half, s + 1]
                    L[j, s + 1] = g_operation(L[j - half, s], L[j, s], top_bit)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block = 1 << s
                half = block >> 1
                for j in range(l, -1, -block):
                    if j % block >= half:
                        B[j - half, s - 1] = B[j, s] ^ B[j - half, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [(1 << i) - 1 for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        if l >= N // 2:
            bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
        else:
            bit_layer_vec.append([])
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Permuted SC，与比特倒序编码器配套）。
    """
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = _channel_llr_layout(llr_ch, N)

    _, llr_layers, bit_layers = precompute_sc_indices(N)

    for phi in range(N):
        l = _bit_reversed(phi, n)
        for s in llr_layers[phi]:
            block = 1 << (s + 1)
            half = block >> 1
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
                else:
                    top_bit = B[j - half, s + 1]
                    L[j, s + 1] = g_operation(L[j - half, s], L[j, s], top_bit)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        for s in bit_layers[phi]:
            block = 1 << s
            half = block >> 1
            for j in range(l, -1, -block):
                if j % block >= half:
                    B[j - half, s - 1] = B[j, s] ^ B[j - half, s]
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
