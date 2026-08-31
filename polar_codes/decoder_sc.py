"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 Permuted SC（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(top, bottom, u) = bottom + (1-2u)*top"""
    return Lb + (1.0 - 2.0 * u_hat) * La


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


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，用于验证）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)

    if N == 1:
        if frozen_bits[0]:
            return np.array([0], dtype=int)
        return np.array([0 if llr[0] >= 0 else 1], dtype=int)

    half = N // 2
    llr_left = f_operation(llr[:half], llr[half:])
    u_left = sc_decode_recursive(llr_left, frozen_bits[:half])
    llr_right = g_operation(llr[:half], llr[half:], u_left)
    u_right = sc_decode_recursive(llr_right, frozen_bits[half:])
    return np.concatenate([u_left, u_right])


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SC 译码（Vangala et al.）。
    信道 LLR 无需比特倒序，按置换顺序逐位译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for i in range(N):
        l = _bit_reversed(i, n)
        start_s = n - _active_llr_level(l, n)
        for s in range(start_s, n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top_llr = L[j, s]
                    btm_llr = L[j + branch_size, s]
                    L[j, s + 1] = f_operation(top_llr, btm_llr)
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N // 2:
            continue
        end_s = n - _active_bit_level(l, n)
        for s in range(n, end_s, -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_nonrecursive_v2(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（供参考）"""
    n = int(np.log2(N))
    lambda_offset = [2 ** (layer - 1) if layer > 0 else 0 for layer in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        layers_llr, temp = [], phi
        while temp % 2 == 1:
            layers_llr.append(int(np.log2(temp & -temp)))
            temp //= 2
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        if phi % 2 == 0:
            psi = phi // 2
            while psi % 2 == 1:
                layers_bit.append(int(np.log2(psi & -psi)))
                psi //= 2
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec
