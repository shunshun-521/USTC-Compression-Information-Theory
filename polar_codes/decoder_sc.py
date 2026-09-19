"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import polar_encode_partial


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    u_hat 为左子树部分编码后的比特（与 La 同形状）
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    n = int(math.log2(N))

    def decode_node(llr_node, depth, bit_offset):
        m = len(llr_node)
        if m == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = m // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, depth - 1, bit_offset)

        u_left = u_hat[bit_offset : bit_offset + half]
        encoded_left = polar_encode_partial(u_left)
        llr_right = g_operation(llr_node[:half], llr_node[half:], encoded_left)
        decode_node(llr_right, depth - 1, bit_offset + half)

    decode_node(llr, n, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layer = 0
        while layer < n and ((phi >> layer) & 1):
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))

        b_layers = []
        for lyr in range(n):
            if (phi >> lyr) & 1:
                b_layers.append(lyr)
        bit_layer_vec.append(b_layers)

    return llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """
    非递归 SC 译码（分层 LLR/比特数组实现，供参考与对比）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[n, :N] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in reversed(llr_layer_vec[phi]):
            psi = (phi >> layer) & 1
            lam = 1 << layer
            if psi == 0:
                for beta in range(lam):
                    P[layer, beta] = f_operation(
                        P[layer + 1, beta], P[layer + 1, beta + lam]
                    )
            else:
                u_partial = u_hat[phi - lam + 1 : phi + 1]
                encoded = polar_encode_partial(u_partial)
                for beta in range(lam):
                    P[layer, beta] = g_operation(
                        P[layer + 1, beta],
                        P[layer + 1, beta + lam],
                        encoded[beta],
                    )

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0, 0] >= 0 else 1

        C[0, 0] = u_hat[phi]
        for layer in bit_layer_vec[phi]:
            lam = 1 << layer
            for beta in range(lam):
                C[layer + 1, beta + lam] = C[layer, beta]
                C[layer + 1, beta] = C[layer + 1, beta + lam] ^ C[layer, beta]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口（递归实现，O(N log N)）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
