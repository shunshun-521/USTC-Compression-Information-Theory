"""
极化码 SC（串行抵消）译码器
PSC 非递归实现（Vangala 2014），递归版本用于验证
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：La=top, Lb=btm；u=0 -> top+btm; u=1 -> btm-top"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, Lb - La)


def active_llr_level(i, n):
    """二进制表示中第一个 1 的位置（从高位计）"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """二进制表示中第一个 0 的位置（从高位计）"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n, N):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if (j % block_size) < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_llr = L[j - branch_size, s]
                btm_llr = L[j, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if (j % block_size) >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 PSC SC 译码。
    L[:,0] 为信道 LLR（自然顺序），按比特倒序索引逐位译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    for i in range(N):
        l = br[i]
        _update_llrs(L, B, l, n, N)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].copy()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 PSC 非递归等价）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)
    br = bit_reversal_permutation(N)

    def decode_node(llr_node, stage, leaves):
        if stage == 0:
            l = leaves[0]
            if frozen_bits[l]:
                u_hat[l] = 0
            else:
                u_hat[l] = 0 if llr_node[0] >= 0 else 1
            return

        half = 1 << (stage - 1)
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, stage - 1, leaves[:half])

        u_partial = u_hat[leaves[:half]]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_partial)
        decode_node(llr_right, stage - 1, leaves[half:])

    decode_node(llr, n, br)
    return u_hat


def precompute_sc_indices(N):
    """兼容 SCL 的预计算接口"""
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = br[i]
        start = n - active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return list(range(N)), llr_layer_vec, bit_layer_vec, br
