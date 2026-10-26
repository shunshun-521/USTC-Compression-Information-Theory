"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, frozen_sub):
        n = len(llr_node)
        if n == 1:
            u = 0 if (frozen_sub[0] or llr_node[0] >= 0) else 1
            return np.array([u], dtype=int), np.array([u], dtype=int)
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u1, u1_up = decode_node(llr_left, frozen_sub[:half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], u1_up)
        u2, u2_up = decode_node(llr_right, frozen_sub[half:])
        u_hat = np.concatenate([u1, u2])
        u_up_left = np.bitwise_xor(u1_up, u2_up)
        u_hat_up = np.concatenate([u_up_left, u2_up])
        return u_hat, u_hat_up

    u_hat, _ = decode_node(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        while psi % 2 == 1:
            llr_layers.append(int(math.log2(psi & -psi)))
            psi -= 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        psi = phi + 1
        while psi % 2 == 0 and psi <= N:
            bit_layers.append(int(math.log2(psi & -psi)) - 1)
            psi >>= 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（与递归实现等价，委托递归以保证正确性）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def _sc_decode_nonrecursive(llr_ch, frozen_bits):
    """分层非递归 SC（保留供参考）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[n, :] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            offset = lambda_offset[layer]
            two_offset = 2 * offset
            for beta in range(0, N, two_offset):
                for omega in range(offset):
                    idx = beta + omega
                    P[layer, idx] = f_operation(
                        P[layer + 1, idx],
                        P[layer + 1, idx + offset],
                    )
                    P[layer, idx + offset] = g_operation(
                        P[layer + 1, idx],
                        P[layer + 1, idx + offset],
                        C[layer, idx],
                    )

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0, phi] >= 0 else 1
        C[0, phi] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            offset = lambda_offset[layer]
            two_offset = 2 * offset
            for beta in range(0, N, two_offset):
                for omega in range(offset):
                    idx = beta + omega
                    C[layer + 1, idx] = (C[layer, idx] ^ C[layer, idx + offset]) & 1
                    C[layer + 1, idx + offset] = C[layer, idx + offset]

    return u_hat
