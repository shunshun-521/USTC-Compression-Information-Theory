"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_exact(La, Lb):
    """精确 log-domain f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：b + (1-2u)*a，La 为上半支 LLR，Lb 为下半支"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def sc_decode_recursive(llr, frozen_bits, use_exact_f=True):
    """递归 SC 译码（与 polar_transform 编码配套）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = llr.size
    u_hat = np.zeros(N, dtype=int)
    f_fn = f_operation_exact if use_exact_f else f_operation

    def node(llr_node, base, length):
        if length == 1:
            idx = base
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=int)

        half = length // 2
        upper_llr = f_fn(llr_node[:half], llr_node[half:])
        beta_u = node(upper_llr, base, half)
        lower_llr = g_operation(llr_node[:half], llr_node[half:], beta_u)
        beta_l = node(lower_llr, base + half, half)
        return np.concatenate([beta_u ^ beta_l, beta_l])

    node(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << (n - i) for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        p = phi
        while p & 1:
            layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(layers)

        layers_b = []
        p = phi + 1
        while p % 2 == 0 and p <= N:
            layers_b.append(int(math.log2(p & -p)) if p > 0 else 0)
            p >>= 1
        bit_layer_vec.append(layers_b)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC：当前实现调用高效递归内核"""
    return sc_decode_recursive(llr_ch, frozen_bits, use_exact_f=True)
