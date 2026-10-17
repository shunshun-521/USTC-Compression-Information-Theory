"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于 LLR 阵列更新）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


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


def f_operation(La, Lb):
    """向量化的 f 运算（精确 boxplus，与 min-sum 接口兼容）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    out = np.empty_like(La)
    it = np.nditer([La, Lb, out], flags=["multi_index"])
    for a, b, o in it:
        o[...] = _upper_llr(float(a), float(b))
    return out


def g_operation(La, Lb, u_hat):
    """g 运算（u_hat 为部分和比特）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=int)
    out = np.empty_like(La)
    for i in range(La.size):
        out[i] = _lower_llr(float(La[i]), float(Lb[i]), int(u_hat[i]))
    return out


def _frozen_mask(frozen_bits):
    return np.asarray(frozen_bits, dtype=bool)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，使用部分和比特）"""

    def f_ms(a, b):
        return _upper_llr(a, b)

    def g_ms(a, b, c):
        return _lower_llr(a, b, c)

    N = len(llr)
    llr = np.asarray(llr, dtype=np.float64)
    frozen = _frozen_mask(frozen_bits)
    decisions = np.zeros(N, dtype=int)

    def sc_sub(llrs, indices):
        if len(indices) == 1:
            i = indices[0]
            if frozen[i]:
                decisions[i] = 0
            else:
                decisions[i] = 0 if llrs[0] >= 0 else 1
            return [decisions[i]]
        half = len(indices) // 2
        left_llr = [f_ms(llrs[i], llrs[i + half]) for i in range(half)]
        enc_left = sc_sub(left_llr, indices[:half])
        right_llr = [g_ms(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)]
        enc_right = sc_sub(right_llr, indices[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    sc_sub(llr, list(range(N)))
    return decisions


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的三个辅助向量（接口兼容）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    lambda_offset = np.zeros(n + 1, dtype=int)
    off = 0
    for layer in range(n + 1):
        lambda_offset[layer] = off
        off += 1 << (n - layer)

    for phi in range(N):
        layers_llr = []
        tmp = phi
        layer = 0
        while tmp % 2 == 1 and layer < n:
            layers_llr.append(layer)
            tmp >>= 1
            layer += 1
        if phi < N - 1:
            layers_llr.append(layer)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        tmp = phi + 1
        layer = 0
        while tmp % 2 == 0 and layer < n:
            layers_bit.append(layer)
            tmp >>= 1
            layer += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（高效阵列实现）。
    frozen_bits: 1/True 表示冻结位。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen = _frozen_mask(frozen_bits)
    frozen_set = set(np.where(frozen)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = np.asarray(llr_ch, dtype=np.float64)

    u_hat = np.zeros(N, dtype=int)

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for l in [_bit_reversed_index(i, n) for i in range(N)]:
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = int(B[l, n])
        update_bits(l)

    return u_hat
