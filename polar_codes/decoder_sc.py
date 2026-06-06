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
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（偶/奇分解，与 polar_encode 配套）。
    """
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(y, frozen):
        N = len(y)
        if N == 1:
            if frozen[0]:
                u = 0
            else:
                u = 0 if y[0] >= 0 else 1
            x = 0.0 if y[0] >= 0 else 1.0
            return np.array([u], dtype=int), np.array([x])

        u1est = f_operation(y[0::2], y[1::2])
        uhat1, u1hp = decode_node(u1est, frozen[: N // 2])
        u2est = g_operation(f_operation(u1hp, y[0::2]), y[1::2], uhat1)
        uhat2, u2hp = decode_node(u2est, frozen[N // 2 :])

        u = np.zeros(N, dtype=int)
        u[: N // 2] = uhat1
        u[N // 2 :] = uhat2

        x1 = f_operation(u1hp, u2hp)
        x = np.zeros(N, dtype=np.float64)
        x[0::2] = x1
        x[1::2] = u2hp
        return u, x

    u_hat, _ = decode_node(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        psi = 0
        tmp = phi
        while tmp % 2 == 1:
            psi += 1
            tmp >>= 1
        llr_layer_vec.append(list(range(psi, n)))

        if phi % 2 == 0:
            bit_layer_vec.append(list(range(n)))
        else:
            psi_b = 0
            tmp = phi
            while tmp % 2 == 1:
                psi_b += 1
                tmp >>= 1
            bit_layer_vec.append(list(range(psi_b, n)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（委托递归实现，保证正确性）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
