"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：下分支 LLR = btm + top (u=0) 或 btm - top (u=1)
    La=top, Lb=btm
    """
    btm = np.asarray(Lb, dtype=np.float64)
    top = np.asarray(La, dtype=np.float64)
    u = np.asarray(u_hat)
    return np.where(u == 0, btm + top, btm - top)


def _bit_reversed_index(i, n):
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= (1 << (n - 1 - b))
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
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    rev = bit_reversal_permutation(N)
    llr = llr[rev]

    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, frozen_node, depth, bit_offset):
        m = len(llr_node)
        if m == 1:
            idx = bit_offset
            if frozen_node[0]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = m // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, frozen_node[:half], depth - 1, bit_offset)

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, frozen_node[half:], depth - 1, bit_offset + half)

    decode_order = [_bit_reversed_index(i, n) for i in range(N)]
    frozen_reordered = np.zeros(N, dtype=bool)
    llr_reordered = np.zeros(N, dtype=np.float64)
    for i, l in enumerate(decode_order):
        frozen_reordered[i] = frozen_bits[l]
        llr_reordered[i] = llr[i]

    decode_node(llr_reordered, frozen_reordered, n, 0)

    result = np.zeros(N, dtype=int)
    for i, l in enumerate(decode_order):
        result[l] = u_hat[i]
    return result


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回 bit-reversed 译码顺序下的层索引。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for i in range(N):
        l = _bit_reversed_index(i, n)
        start_s = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start_s, n)))

        start_bit_s = n - _active_bit_level(l, n) + 1
        bit_layer_vec.append(list(range(n, start_bit_s - 1, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（基于因子图树的高效实现）。
    信道 LLR 在输入时做比特倒序，与编码器输出倒序一致。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    rev = bit_reversal_permutation(N)
    llr_ch = llr_ch[rev]
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = llr_ch

    for i in range(N):
        l = _bit_reversed_index(i, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s],
                        int(B[j - branch_size, s + 1])
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size >> 1
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
