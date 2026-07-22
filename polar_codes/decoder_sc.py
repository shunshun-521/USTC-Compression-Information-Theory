"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


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
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码。
    """
    N = len(llr)
    brp = bit_reversal_permutation(N)
    llr = np.asarray(llr, dtype=np.float64)[brp]
    u_hat = np.zeros(N, dtype=int)
    frozen_bits = np.asarray(frozen_bits, dtype=int)

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
            decode_node(llr_left[i:i + 1], bit_offset + i)

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i:i + 1], bit_offset + half + i)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        temp = phi
        for layer in range(n):
            if (temp & 1) == 0:
                layers_llr.append(layer)
            else:
                break
            temp >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        temp = phi + 1
        for layer in range(n):
            if (temp & 1) == 1:
                layers_bit.append(layer)
            else:
                break
            temp >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（调用高效分层实现）。
    """
    return sc_decode_efficient(llr_ch, frozen_bits)


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


def sc_decode_efficient(llr_ch, frozen_bits):
    """
    非递归 SC 译码（分层 LLR/比特数组实现）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    brp = bit_reversal_permutation(N)
    llr_ch = llr_ch[brp]
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=int)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    for i in range(N):
        l = _bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
