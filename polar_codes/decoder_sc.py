"""
极化码 SC（串行抵消）译码器
置换 SC（Vangala et al.）+ 递归参考实现
"""
import math
import numpy as np
from encoder import bit_reversed


LLR_CLIP = 30.0


def _clip(x):
    return np.clip(x, -LLR_CLIP, LLR_CLIP)


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    """f 运算（对数域 boxplus）。"""
    l1, l2 = _clip(l1), _clip(l2)
    return _clip(logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2))


def lower_llr(l1, l2, b):
    """g 运算：l1=下分支 LLR，l2=上分支 LLR。"""
    l1, l2 = _clip(l1), _clip(l2)
    b = np.asarray(b)
    if b.ndim == 0 and np.isnan(b):
        b = 0
    b = np.nan_to_num(b, nan=0.0)
    return _clip(np.where(b == 0, l1 + l2, l1 - l2))


def f_operation(La, Lb):
    """min-sum 近似 f（供 BP 等模块复用）。"""
    return _clip(
        np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))
    )


def g_operation(La, Lb, u_hat):
    """g 运算：La=上分支，Lb=下分支。"""
    return lower_llr(Lb, La, u_hat)


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
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = lower_llr(
                    L[j, s], L[j - branch_size, s], top_bit
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（自然顺序，仅作参考）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        if len(llr_node) == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return
        half = len(llr_node) // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        for i in range(half):
            decode_node(llr_left[i : i + 1], bit_offset + i)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i : i + 1], bit_offset + half + i)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(
            [] if l < N // 2 else list(range(n, n - _active_bit_level(l, n), -1))
        )
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归置换 SC 译码。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits == 1)[0])

    for phi in range(N):
        l = bit_reversed(phi, n)
        _update_llrs(L, B, l, n, N)

        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        _update_bits(B, l, n, N)

    return u_hat
