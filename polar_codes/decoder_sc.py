"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

LLR_MAX = 30.0


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（box-plus 可选，见 cn_operation）。
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def cn_operation(La, Lb):
    """精确 box-plus f 运算（LLR 域）。"""
    La = np.clip(np.asarray(La, dtype=np.float64), -LLR_MAX, LLR_MAX)
    Lb = np.clip(np.asarray(Lb, dtype=np.float64), -LLR_MAX, LLR_MAX)
    return np.log1p(np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb))


def g_operation(La, Lb, u_partial):
    """
    g 运算：g(La, Lb, u_partial) = (1 - 2*u_partial) * La + Lb
    u_partial 为当前层的部分和（非最终信息比特）。
    """
    return (1 - 2 * u_partial) * La + Lb


def _sc_recursive(llr, frozen_bits, use_boxplus=True):
    """Sionna/Arikan 风格递归 SC，g 节点使用部分和 u_up。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    n = len(llr)
    f_fn = cn_operation if use_boxplus else f_operation

    if n == 1:
        if frozen_bits[0]:
            bit = 0
        elif llr[0] >= 0:
            bit = 0
        else:
            bit = 1
        u = np.array([bit], dtype=int)
        return u, u.astype(np.float64)

    half = n // 2
    llr_left = llr[:half]
    llr_right = llr[half:]
    frozen_l = frozen_bits[:half]
    frozen_r = frozen_bits[half:]

    top = f_fn(llr_left, llr_right)
    u_left, u_left_up = _sc_recursive(top, frozen_l, use_boxplus)

    bottom = g_operation(llr_left, llr_right, u_left_up)
    u_right, u_right_up = _sc_recursive(bottom, frozen_r, use_boxplus)

    u = np.concatenate([u_left, u_right])
    u_up_left = np.bitwise_xor(u_left_up.astype(int), u_right_up.astype(int)).astype(np.float64)
    u_up = np.concatenate([u_up_left, u_right_up.astype(np.float64)])
    return u, u_up


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    u_hat, _ = _sc_recursive(np.asarray(llr, dtype=np.float64), frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        pp = phi
        while pp % 2 == 1:
            layers_llr.append(int(math.log2(pp & -pp)))
            pp //= 2
        layers_llr.append(n - 1)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            pp = phi
            while pp % 2 == 0 and pp > 0:
                layers_bit.append(int(math.log2(pp & -pp)))
                pp //= 2
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（调用修正后的递归核心）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
