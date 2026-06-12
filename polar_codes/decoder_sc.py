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
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
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
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        node_len = len(llr_node)
        if node_len == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = node_len // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        psi = phi
        layer = 0
        while psi % 2 == 1:
            llr_layers.append(layer)
            psi //= 2
            layer += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 1:
            psi = phi
            layer = 0
            while psi % 2 == 1:
                bit_layers.append(layer)
                psi //= 2
                layer += 1
        bit_layer_vec.append(bit_layers)

    lambda_offset = [1 << layer for layer in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Permuted SCD，与编码器配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    frozen_indices = set(np.where(frozen_bits)[0])
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for bit_idx in [_bit_reversed(i, n) for i in range(N)]:
        for stage in range(n - _active_llr_level(bit_idx, n), n):
            block_size = 2 ** (stage + 1)
            branch_size = block_size // 2
            for j in range(bit_idx, N, block_size):
                if j % block_size < branch_size:
                    L[j, stage + 1] = f_operation(L[j, stage], L[j + branch_size, stage])
                else:
                    L[j, stage + 1] = g_operation(
                        L[j - branch_size, stage],
                        L[j, stage],
                        B[j - branch_size, stage + 1],
                    )

        if bit_idx in frozen_indices:
            B[bit_idx, n] = 0
        else:
            B[bit_idx, n] = 0 if L[bit_idx, n] >= 0 else 1

        if bit_idx < N // 2:
            continue

        for stage in range(n, n - _active_bit_level(bit_idx, n), -1):
            block_size = 2 ** stage
            branch_size = block_size // 2
            for j in range(bit_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, stage - 1] = int(B[j, stage]) ^ int(B[j - branch_size, stage])
                    B[j, stage - 1] = B[j, stage]

    return B[:, n].astype(int)
