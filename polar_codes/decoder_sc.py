"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """精确 log 域 f 运算（check node）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    return Lb + (1.0 - 2.0 * u_hat.astype(np.float64)) * La


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与极化树结构一致）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.uint8)

    def leaf(idx, llr_val):
        if frozen_bits[idx]:
            u_hat[idx] = 0
        else:
            u_hat[idx] = 0 if llr_val >= 0 else 1
        return np.array([u_hat[idx]], dtype=np.uint8)

    def node(channel_llr, base, length):
        if length == 1:
            leaf(base, float(channel_llr[0]))
            return np.array([u_hat[base]], dtype=np.uint8)

        half = length // 2
        upper = f_operation(channel_llr[:half], channel_llr[half:])
        beta_u = node(upper, base, half)
        lower = g_operation(channel_llr[:half], channel_llr[half:], beta_u)
        beta_l = node(lower, base + half, half)
        return np.concatenate([beta_u ^ beta_l, beta_l])

    node(llr, 0, N)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [2**i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        if phi == 0:
            llr_layer_vec.append(list(range(n)))
        else:
            layers_llr = []
            tmp = phi
            layer = 0
            while tmp % 2 == 0 and layer < n:
                layers_llr.append(layer)
                tmp //= 2
                layer += 1
            llr_layer_vec.append(layers_llr)

        layers_bit = []
        tmp = phi + 1
        layer = 0
        while tmp % 2 == 0 and layer < n:
            layers_bit.append(layer)
            tmp //= 2
            layer += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（P/C 数组实现，供对照）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    _, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[n, :] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            sp = 2**layer
            for j in range(0, N, 2 * sp):
                for jj in range(j, j + sp):
                    P[layer, jj] = f_operation(
                        P[layer + 1, jj], P[layer + 1, jj + sp]
                    )
                    P[layer, jj + sp] = g_operation(
                        P[layer + 1, jj],
                        P[layer + 1, jj + sp],
                        C[layer, jj],
                    )

        if frozen_bits[phi]:
            u_hat[phi] = 0
            C[0, phi] = 0
        else:
            u_hat[phi] = 0 if P[0, phi] >= 0 else 1
            C[0, phi] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            sp = 2**layer
            for j in range(0, N, 2 * sp):
                for jj in range(j, j + sp):
                    C[layer + 1, jj + sp] = C[layer, jj] ^ C[layer, jj + sp]
                    C[layer + 1, jj] = C[layer, jj]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（当前采用递归树实现，与列表大小为 1 的 SCL 等价）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
