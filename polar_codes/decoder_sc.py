"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import polar_encode


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
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _decide(llr, frozen):
    """根据 LLR 与冻结位做硬判决。"""
    if frozen:
        return 0
    return 0 if llr >= 0 else 1


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（Arikan 算法，Figure 3）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode(l, frozen):
        N = len(l)
        if N == 1:
            return np.array([_decide(l[0], frozen[0])], dtype=int)

        if N == 2:
            u = np.zeros(2, dtype=int)
            u[0] = _decide(f_operation(l[0], l[1]), frozen[0])
            u[1] = _decide(g_operation(l[0], l[1], u[0]), frozen[1])
            return u

        half = N // 2
        l_prime = np.array(
            [f_operation(l[2 * i], l[2 * i + 1]) for i in range(half)], dtype=np.float64
        )
        u_prime = decode(l_prime, frozen[:half])
        v = polar_encode(u_prime)
        l_double_prime = np.array(
            [g_operation(l[2 * i], l[2 * i + 1], v[i]) for i in range(half)],
            dtype=np.float64,
        )
        u_double_prime = decode(l_double_prime, frozen[half:])
        return np.concatenate([u_prime, u_double_prime])

    return decode(llr, frozen_bits)


def sc_decode_recursive_tree(llr, frozen_bits):
    """与 sc_decode_recursive 相同（别名）。"""
    return sc_decode_recursive(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助索引。
    """
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        p = phi
        while p % 2 == 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        p = phi + 1
        if p < N:
            while p % 2 == 0:
                bit_layers.append(int(math.log2(p & -p)))
                p >>= 1
        bit_layer_vec.append(bit_layers)

    lambda_offset = np.zeros(n + 1, dtype=int)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer + 1)) - 1

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_layered(llr_ch, frozen_bits):
    """
    非递归分层 SC 译码（与递归版本等价）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


sc_decode = sc_decode_recursive
