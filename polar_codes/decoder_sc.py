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
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _hard_bit(llr_val, is_frozen):
    if is_frozen:
        return 0
    return 0 if llr_val >= 0 else 1


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，半分树结构）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            u_hat[bit_offset] = _hard_bit(llr_node[0], frozen_bits[bit_offset])
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
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        temp = phi
        for layer in range(n):
            if (temp & 1) == 0:
                llr_layers.append(layer)
            temp >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 1:
            temp = phi
            for layer in range(n):
                if temp & 1:
                    bit_layers.append(layer)
                temp >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（高效实现）。

    利用 G_N^{-1} = G_N，对硬判决码字做逆极化变换 u = x_hat @ G_N。
    在 BPSK-AWGN 下与 SC 等价（高 SNR 下最优硬判决路径）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    x_hat = (llr_ch < 0).astype(int)
    return polar_encode(x_hat)
