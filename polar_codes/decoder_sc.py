"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效 Permuted SCD 实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * np.asarray(u_hat, dtype=np.float64)) * La + Lb


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


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
    """递归 SC 译码（参考实现）"""
    from channel import reorder_channel_llr

    llr = reorder_channel_llr(np.asarray(llr, dtype=np.float64))
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
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容 SCL 接口）"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = 2 ** (i - 1)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))

        if l < N // 2:
            bit_layer_vec.append([])
        else:
            end = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, end, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs_pscd(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, len(L), block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                if np.isnan(top_bit):
                    top_bit = 0
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], top_bit
                )


def _update_bits_pscd(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                bj = 0 if np.isnan(B[j, s]) else int(B[j, s])
                bjb = 0 if np.isnan(B[j - branch_size, s]) else int(B[j - branch_size, s])
                B[j - branch_size, s - 1] = bj ^ bjb
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SC 译码。
  信道 LLR 经比特倒序重排后与编码端 B_N 一致。
    """
    from channel import reorder_channel_llr

    llr_ch = reorder_channel_llr(np.asarray(llr_ch, dtype=np.float64))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch

    for phi in range(N):
        l = _bit_reversed(phi, n)
        _update_llrs_pscd(L, B, l, n)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits_pscd(B, l, n, N)

    u_hat = np.nan_to_num(B[:, n], nan=0.0).astype(int)
    return u_hat


def sc_decode_with_reorder(llr_ch, frozen_bits):
    """对信道 LLR 执行 SC 译码"""
    return sc_decode(llr_ch, frozen_bits)
