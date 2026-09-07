"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于 polarcodes SCD）
"""
import math

import numpy as np

from encoder import bit_reversed


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = La + (1 - 2*u_hat) * Lb（lower_llr 形式）。"""
    return La + (1 - 2 * u_hat) * Lb


def upper_llr(top, btm):
    return f_operation(top, btm)


def lower_llr(btm, top, bit):
    return g_operation(btm, top, bit)


def active_llr_level(i, n):
    """找到索引 i 的二进制表示中第一个 1 的位置（从高位计）。"""
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
    """找到索引 i 的二进制表示中第一个 0 的位置（从高位计）。"""
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
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return
        half = n // 2
        top, btm = llr_node[:half], llr_node[half:]
        llr_left = f_operation(top, btm)
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = lower_llr(btm, top, u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def sc_decode_core(llr_ch, frozen_bits):
    """
    非递归 SC 译码（polarcodes SCD 风格）。
    信道 LLR 置于 L[:,0]，在比特倒序索引顺序下逐位译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed(i, n)
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return sc_decode_core(llr_ch, frozen_bits)
