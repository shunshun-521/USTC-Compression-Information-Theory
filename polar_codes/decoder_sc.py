"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Vangala PSC）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(x, n):
    """与 encoder 一致的比特倒序"""
    return int(format(x, f'0{n}b')[::-1], 2)


def _active_llr_level(i, n):
    """从最高位起，第一个 0 之前已检查的层数"""
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
    """从最高位起，第一个 1 之前已检查的层数"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n):
    """更新第 l 个信息位所需的 LLR 树"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, len(L), block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s],
                    L[j, s],
                    B[j - branch_size, s + 1],
                )


def _update_bits(B, l, n):
    """比特回传"""
    if l < len(B) // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Vangala 置换 PSC，min-sum）。
    信道 LLR 应已通过 channel_llr() 做比特倒序对齐。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = _bit_reversed(i, n)
        _update_llrs(L, B, l, n)

        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit

        _update_bits(B, l, n)

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，min-sum）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        for i in range(half):
            decode_node(llr_left[i : i + 1], bit_offset + i)

        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i : i + 1], bit_offset + half + i)

    n = int(math.log2(N))
    order = [_bit_reversed(i, n) for i in range(N)]
    llr_perm = np.zeros(N, dtype=np.float64)
    for i, l in enumerate(order):
        llr_perm[l] = llr[i]
    decode_node(llr_perm, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算辅助向量（供文档/扩展使用）"""
    n = int(math.log2(N))
    lambda_offset = []
    llr_layer_vec = []
    bit_layer_vec = []

    for i in range(N):
        l = _bit_reversed(i, n)
        llr_start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(llr_start, n)))

        bit_start = n - _active_bit_level(l, n)
        if l >= N // 2:
            bit_layer_vec.append(list(range(n, bit_start, -1)))
        else:
            bit_layer_vec.append([])

        lambda_offset.append(1 << i)

    br = bit_reversal_permutation(N)
    return lambda_offset, llr_layer_vec, bit_layer_vec, br
