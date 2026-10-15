"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于分层 LLR 更新）
"""
import math
import numpy as np


def bit_reversed(x, n):
    """比特倒序（单索引）"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（支持向量化）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：La 为上半支路 LLR，Lb 为下半支路 LLR"""
    return (1 - 2 * u_hat) * La + Lb


def _upper_llr(l1, l2):
    """上半支路 LLR 更新（标量 min-sum）"""
    return float(f_operation(l1, l2))


def _lower_llr(l1, l2, b):
    """下半支路 LLR 更新（l1=下支路，l2=上支路）"""
    if b == 0:
        return l1 + l2
    return l1 - l2


def _active_llr_level(i, n):
    """二进制表示中首个 1 的位置（从高位计）"""
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
    """二进制表示中首个 0 的位置（从高位计）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                top_bit = int(B[j - branch_size, s + 1])
                L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)


def _update_bits(B, l, n):
    if l < B.shape[0] // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    frozen_bits: True/1 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    frozen_set = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = bit_reversed(i, n)
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, n].astype(int)


def _decode_chunk(llr_node, bit_offset, depth, frozen_bits, u_hat):
    """递归 SC（参考实现）"""
    if depth == 0:
        idx = bit_offset
        if frozen_bits[idx]:
            u_hat[idx] = 0
        else:
            u_hat[idx] = 0 if llr_node[0] >= 0 else 1
        return

    half_len = 1 << (depth - 1)
    llr_left = f_operation(llr_node[:half_len], llr_node[half_len : 2 * half_len])
    _decode_chunk(llr_left, bit_offset, depth - 1, frozen_bits, u_hat)

    u_left = u_hat[bit_offset : bit_offset + half_len]
    llr_right = g_operation(
        llr_node[:half_len], llr_node[half_len : 2 * half_len], u_left
    )
    _decode_chunk(llr_right, bit_offset + half_len, depth - 1, frozen_bits, u_hat)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价，用于交叉验证）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    depth = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)
    _decode_chunk(llr, 0, depth, frozen_bits, u_hat)
    return u_hat


def precompute_sc_indices(N):
    """保留接口：预计算向量（供扩展/调试）"""
    n = int(math.log2(N))
    lambda_offset = [2 ** (layer - 1) if layer > 0 else 0 for layer in range(n + 1)]
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
