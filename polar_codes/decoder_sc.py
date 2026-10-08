"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
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
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        temp = phi
        for layer in range(n):
            if (temp & 1) == 0:
                layers_llr.append(layer)
            temp >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        temp = phi
        for layer in range(n):
            if (temp & 1) == 1:
                layers_bit.append(layer)
            temp >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（与递归版本等价性以 validate 为准）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    _, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    u_hat = np.zeros(N, dtype=int)

    P[n, :] = llr_ch

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            spm = 1 << layer
            left = (phi >> (layer + 1)) << (layer + 1)
            right = left + spm
            node = left >> layer
            P[layer, node] = f_operation(P[layer + 1, left], P[layer + 1, right])
            P[layer, node + 1] = g_operation(
                P[layer + 1, left], P[layer + 1, right], C[layer, node]
            )

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0, 0] >= 0 else 1

        C[0, 0] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            spm = 1 << layer
            left = (phi >> (layer + 1)) << (layer + 1)
            right = left + spm
            node = left >> layer
            C[layer + 1, right] = C[layer, node] ^ C[layer, node + 1]
            C[layer + 1, left] = C[layer, node + 1]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口（与递归实现一致）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
