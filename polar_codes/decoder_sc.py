"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversed_index


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


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


def _update_llrs(L, B, l, n, N):
    """增量更新 LLR 树"""
    start = n - _active_llr_level(l, n)
    for s in range(start, n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                top = L[j, s]
                btm = L[j + branch_size, s]
                L[j, s + 1] = f_operation(top, btm)
            else:
                btm = L[j, s]
                top = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(top, btm, top_bit)


def _update_bits(B, l, n, N):
    """比特回传"""
    if l < N // 2:
        return
    end = n - _active_bit_level(l, n)
    for s in range(n, end, -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    frozen_bits: 1/True 表示冻结位
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)
    decode_order = [bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        _update_llrs(L, B, l, n, N)
        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]
        _update_bits(B, l, n, N)

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = int(math.log2(N))

    def decode_node(llr_node, bit_indices):
        m = len(llr_node)
        if m == 1:
            idx = bit_indices[0]
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = m // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_indices[:half])

        u_left = u_hat[bit_indices[:half]]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_indices[half:])

    order = [bit_reversed_index(i, n) for i in range(N)]
    inv_order = np.argsort(order)
    llr_perm = llr[inv_order]
    decode_node(llr_perm, order)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for phi in range(N):
        t = phi
        while t > 0 and (t & 1):
            llr_layer_vec[phi].append((t & -t).bit_length() - 1)
            t >>= 1
        t = phi
        while t > 0 and not (t & 1):
            bit_layer_vec[phi].append((t & -t).bit_length() - 1)
            t >>= 1
    return lambda_offset, llr_layer_vec, bit_layer_vec
