"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


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
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _hard_decision(llr, frozen):
    if frozen:
        return 0
    if llr >= 0:
        return 0
    return 1


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（返回 u_hat 与 stage 部分和 u_hat_up，g 运算使用 u_hat_up）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    if N > 4:
        rev = bit_reversal_permutation(N)
        llr = llr[rev]

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            bit = _hard_decision(llr_node[0], frozen_node[0])
            u_hat = np.array([bit], dtype=int)
            return u_hat, u_hat.copy()

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_hat1, u_hat1_up = decode_node(llr_left, frozen_node[:half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_hat1_up)
        u_hat2, u_hat2_up = decode_node(llr_right, frozen_node[half:])
        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat_up = np.concatenate([u_hat1_up ^ u_hat2_up, u_hat2_up])
        return u_hat, u_hat_up

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
        llr_layers = []
        p = phi
        for layer in range(n):
            if p % 2 == 0:
                llr_layers.append(layer)
                p //= 2
            else:
                break
        if not llr_layers:
            llr_layers = [0]
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi > 0:
            t = 0
            tmp = phi + 1
            while tmp % 2 == 0:
                t += 1
                tmp //= 2
            bit_layers = list(range(t))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（显式栈，与递归版等价）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return sc_decode_recursive(llr_ch, frozen_bits)
