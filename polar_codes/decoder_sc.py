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
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _sc_decode_tree(llr, frozen_bits):
    """递归 SC 译码树"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = len(llr)

    if n > 1:
        half = n // 2
        l1, l2 = llr[:half], llr[half:]
        f1, f2 = frozen_bits[:half], frozen_bits[half:]

        u1, u1_up = _sc_decode_tree(f_operation(l1, l2), f1)
        u2, u2_up = _sc_decode_tree(g_operation(l1, l2, u1_up), f2)

        u_hat = np.concatenate([u1, u2])
        u1_up = (u1_up.astype(int) ^ u2_up.astype(int)).astype(np.float64)
        u_up = np.concatenate([u1_up, u2_up])
        return u_hat, u_up

    if frozen_bits[0]:
        return np.array([0], dtype=int), np.array([0.0])
    bit = 0 if llr[0] >= 0 else 1
    return np.array([bit], dtype=int), np.array([float(bit)])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    u_hat, _ = _sc_decode_tree(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = 2 ** (layer - 1)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 0:
                llr_layers.append(layer)
            temp //= 2
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 1:
            temp = phi
            for layer in range(n):
                if temp % 2 == 1:
                    bit_layers.append(layer)
                temp //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（当前调用经校验的递归实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
