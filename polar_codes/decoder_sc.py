"""
极化码 SC（串行抵消）译码器
基于 Vangala 置换 SC 算法，使用精确 box-plus LLR 运算
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def _logdomain_sum(a, b):
    max_val = max(a, b)
    if not np.isfinite(max_val):
        return max_val
    return max_val + np.log1p(np.exp(-abs(a - b)))


def upper_llr(l1, l2):
    """f 运算（box-plus）"""
    if np.isnan(l1) or np.isnan(l2):
        return np.nan
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(btm, top, bit):
    """g 运算：btm 为下分支 LLR，top 为上分支 LLR"""
    if np.isnan(btm) or np.isnan(top) or np.isnan(bit):
        return np.nan
    bit = int(bit)
    if bit == 0:
        return btm + top
    return btm - top


def f_operation(La, Lb):
    """向量化 f 运算（供 BP 使用）"""
    return upper_llr(La, Lb)


def g_operation(La, Lb, u_hat):
    """向量化 g 运算：La=上分支, Lb=下分支"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, Lb + La, Lb - La)


def _bit_reversed(i, n):
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
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


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（自然顺序，参考实现）"""
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)
    pos = [0]

    def decode_node(llr_node, length):
        if length == 1:
            idx = pos[0]
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            pos[0] += 1
            return
        half = length // 2
        left = np.array([upper_llr(llr_node[i], llr_node[i + half]) for i in range(half)])
        decode_node(left, half)
        right = np.array(
            [
                lower_llr(llr_node[i + half], llr_node[i], u_hat[pos[0] - half + i])
                for i in range(half)
            ]
        )
        decode_node(right, half)

    decode_node(np.asarray(llr, dtype=np.float64), N)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """置换 SC 译码（Vangala 算法）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = _bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            half = block >> 1
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + half, s])
                else:
                    top_bit = 0 if np.isnan(B[j - half, s + 1]) else int(B[j - half, s + 1])
                    L[j, s + 1] = lower_llr(L[j, s], L[j - half, s], top_bit)

        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
            u_hat[l] = int(B[l, n])

        if l < N // 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block = 1 << s
            half = block >> 1
            for j in range(l, -1, -block):
                if j % block >= half:
                    prev = 0 if np.isnan(B[j - half, s]) else int(B[j - half, s])
                    B[j - half, s - 1] = int(B[j, s]) ^ prev
                    B[j, s - 1] = B[j, s]

    return u_hat


def precompute_sc_indices(N):
    """预计算辅助信息（供 SCL 使用）"""
    n = int(math.log2(N))
    return bit_reversal_permutation(N), [1 << i for i in range(n + 1)], [], []
