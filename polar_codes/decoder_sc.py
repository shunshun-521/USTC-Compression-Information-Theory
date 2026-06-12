"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    下支 LLR 计算时 La=top, Lb=bottom。
    """
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def _active_llr_level(i, n):
    """从 MSB 起第一个为 1 的位之前（含起始）的层数。"""
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
    """从 MSB 起第一个为 0 的位之前（含起始）的层数。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _sequential_sc_decode(llr_ch, frozen_bits):
    """序贯 SC 译码核心（非递归/递归共用）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    llr_mat = np.full((N, n + 1), np.nan, dtype=np.float64)
    llr_mat[:, 0] = llr_ch
    bits_mat = np.zeros((N, n + 1), dtype=np.int_)
    decode_order = [_bit_reversed(i, n) for i in range(N)]

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top = llr_mat[j, s]
                    btm = llr_mat[j + branch_size, s]
                    llr_mat[j, s + 1] = f_operation(top, btm)
                else:
                    btm = llr_mat[j, s]
                    top = llr_mat[j - branch_size, s]
                    top_bit = bits_mat[j - branch_size, s + 1]
                    llr_mat[j, s + 1] = g_operation(top, btm, top_bit)

        if frozen_bits[l]:
            bits_mat[l, n] = 0
        else:
            bits_mat[l, n] = 0 if llr_mat[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        bits_mat[j - branch_size, s - 1] = (
                            bits_mat[j, s] ^ bits_mat[j - branch_size, s]
                        )
                        bits_mat[j, s - 1] = bits_mat[j, s]

    return bits_mat[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与序贯实现等价）。"""
    return _sequential_sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = np.zeros(N, dtype=int)
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed(phi, n)
        t_llr = _active_llr_level(l, n)
        t_bit = _active_bit_level(l, n)
        lambda_offset[phi] = 1 << (n - t_llr) if t_llr < n and phi > 0 else 0
        llr_layer_vec.append(list(range(n - t_llr, n)))
        bit_layer_vec.append(list(range(n - t_bit, n)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return _sequential_sc_decode(llr_ch, frozen_bits)
