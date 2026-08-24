"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversed


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def active_llr_level(i, n):
    """从最高位开始找第一个 1 之前 0 的个数（mcba1n 约定）。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """从最高位开始找第一个 0 之前 1 的个数。"""
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
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def decode_block(llr_block, frozen_block, offset):
        block_len = len(llr_block)
        if block_len == 1:
            if frozen_block[0]:
                u_hat[offset] = 0
            else:
                u_hat[offset] = 0 if llr_block[0] >= 0 else 1
            return

        half = block_len // 2
        llr_left = f_operation(llr_block[:half], llr_block[half:])
        decode_block(llr_left, frozen_block[:half], offset)
        u_left = u_hat[offset:offset + half]
        llr_right = g_operation(llr_block[:half], llr_block[half:], u_left)
        decode_block(llr_right, frozen_block[half:], offset + half)

    decode_block(llr, frozen_bits, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（mcba1n 风格，比特倒序相位顺序）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    frozen_set = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = bit_reversed(i, n)

        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top_llr = L[j, s]
                    btm_llr = L[j + branch_size, s]
                    L[j, s + 1] = f_operation(top_llr, btm_llr)
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = (
                            B[j, s] ^ B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    return B[:, n]


def precompute_sc_indices(N):
    """保留接口：返回比特倒序相位顺序的层索引。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed(i, n)
        llr_layers = list(range(n - active_llr_level(l, n), n))
        bit_layers = list(range(n, n - active_bit_level(l, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
