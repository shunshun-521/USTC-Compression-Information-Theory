"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _sc_decode_core(llr_node, frozen_node):
    """递归 SC 核心，返回 (u_hat, u_hat_up)。"""
    n = len(llr_node)
    if n == 1:
        if frozen_node[0]:
            u = 0
        else:
            u = 0 if llr_node[0] >= 0 else 1
        u = int(u)
        return np.array([u], dtype=int), np.array([u], dtype=np.float64)

    half = n // 2
    llr1 = llr_node[:half]
    llr2 = llr_node[half:]
    frozen1 = frozen_node[:half]
    frozen2 = frozen_node[half:]

    u_hat1, u_hat1_up = _sc_decode_core(
        f_operation(llr1, llr2), frozen1
    )
    llr2_in = g_operation(llr1, llr2, u_hat1_up)
    u_hat2, u_hat2_up = _sc_decode_core(llr2_in, frozen2)

    u_hat = np.concatenate([u_hat1, u_hat2])
    u_up_left = np.bitwise_xor(
        u_hat1_up.astype(np.int64), u_hat2_up.astype(np.int64)
    ).astype(np.float64)
    u_hat_up = np.concatenate([u_up_left, u_hat2_up])
    return u_hat, u_hat_up


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat, _ = _sc_decode_core(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量。"""
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        if phi == 0:
            llr_layer_vec.append(list(range(n)))
        else:
            layers = []
            p = phi
            layer = 0
            while p % 2 == 0:
                layers.append(layer)
                p //= 2
                layer += 1
            llr_layer_vec.append(layers)

        bit_layers = []
        p = phi
        layer = 0
        while p % 2 == 1:
            bit_layers.append(layer)
            p //= 2
            layer += 1
        bit_layer_vec.append(bit_layers)

    return list(range(N)), llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（层矩阵）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.float64)
    P[n, :] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    _, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            step = 1 << (n - 1 - layer)
            for i in range(0, N, 2 * step):
                La = P[layer + 1, i: i + step]
                Lb = P[layer + 1, i + step: i + 2 * step]
                P[layer, i: i + step] = f_operation(La, Lb)

        if phi > 0:
            p = phi
            l = 0
            while p % 2 == 0:
                p //= 2
                l += 1
            for layer in range(l - 1, -1, -1):
                step = 1 << (n - 1 - layer)
                for i in range(0, N, 2 * step):
                    La = P[layer + 1, i: i + step]
                    Lb = P[layer + 1, i + step: i + 2 * step]
                    u_partial = C[layer + 1, i: i + step].astype(np.float64)
                    P[layer, i + step: i + 2 * step] = g_operation(La, Lb, u_partial)

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0, 0] >= 0 else 1

        C[0, 0] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            step = 1 << (n - 1 - layer)
            for i in range(0, N, 2 * step):
                C[layer, i] = np.mod(C[layer + 1, i] + C[layer + 1, i + step], 2)
                C[layer, i + step] = C[layer + 1, i + step]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 主入口（默认调用已验证的递归实现）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
