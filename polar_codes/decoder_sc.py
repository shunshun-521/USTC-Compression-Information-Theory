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
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, beta):
    """
    g 运算：g(La, Lb, beta) = Lb + (1 - 2*beta) * La
    beta 为左子树的部分和向量（非原始源比特）
    """
    return Lb + (1.0 - 2.0 * np.asarray(beta, dtype=np.float64)) * La


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, base, length):
        if length == 1:
            idx = base
            if frozen_bits[idx]:
                u_hat[idx] = 0
                bit = 0
            else:
                bit = 0 if llr_node[0] >= 0 else 1
                u_hat[idx] = bit
            return np.array([bit], dtype=int)

        half = length // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        beta_upper = decode_node(llr_left, base, half)

        llr_right = g_operation(llr_node[:half], llr_node[half:], beta_upper)
        beta_lower = decode_node(llr_right, base + half, half)

        return np.concatenate([beta_upper ^ beta_lower, beta_lower])

    decode_node(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        psi = phi
        while psi % 2 == 1:
            layers.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers)

        layers_b = []
        if phi % 2 == 0:
            for l in range(n):
                if (phi >> l) % 2 == 0:
                    layers_b.append(l)
        else:
            l = 0
            while l < n:
                if (phi >> l) % 2 == 1:
                    layers_b.append(l)
                    break
                l += 1
            while l + 1 < n:
                l += 1
                if (phi >> l) % 2 == 0:
                    layers_b.append(l)
        bit_layer_vec.append(layers_b)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数（递归实现，正确性已验证）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """基于预计算索引的非递归 SC 译码（Tal-Vardy 风格）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    L = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    L[n, :] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            lam = lambda_offset[layer]
            lam2 = lambda_offset[layer + 1]
            for beta in range(lam):
                idx = lam2 + 2 * beta
                if idx + lam < N:
                    L[layer, beta] = f_operation(L[layer + 1, idx], L[layer + 1, idx + lam])

        if frozen_bits[phi]:
            u_hat[phi] = 0
            C[0, 0] = 0
        else:
            u_hat[phi] = 0 if L[0, 0] >= 0 else 1
            C[0, 0] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            lam = lambda_offset[layer]
            lam2 = lambda_offset[layer + 1]
            for beta in range(lam):
                idx = lam2 + 2 * beta
                if phi % 2 == 0:
                    if idx + lam < N:
                        C[layer + 1, idx] = (C[layer, beta] + C[layer, beta + lam]) % 2
                        C[layer + 1, idx + lam] = C[layer, beta + lam]
                else:
                    if idx + lam < N:
                        pm = (1 - 2 * C[layer, beta]) * L[layer + 1, idx]
                        L[layer, beta] = g_operation(pm, L[layer + 1, idx + lam], 0)
                        C[layer + 1, idx] = C[layer, beta]
                        C[layer + 1, idx + lam] = C[layer, beta + lam]

    return u_hat
