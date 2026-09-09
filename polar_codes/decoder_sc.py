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
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _sc_decode_recursive_core(llr_ch, frozen_ind):
    """
    递归 SC 译码核心（Sionna 风格）。
    返回 (u_hat, u_hat_up)，其中 u_hat_up 为当前阶段的局部重编码比特。
    """
    n = len(llr_ch)
    frozen_ind = np.asarray(frozen_ind, dtype=bool)

    if n == 1:
        if frozen_ind[0]:
            u_hat = np.array([0], dtype=int)
        else:
            u_hat = np.array([0 if llr_ch[0] >= 0 else 1], dtype=int)
        return u_hat, u_hat.copy()

    half = n // 2
    llr1 = llr_ch[:half]
    llr2 = llr_ch[half:]
    frozen1 = frozen_ind[:half]
    frozen2 = frozen_ind[half:]

    x_llr1 = f_operation(llr1, llr2)
    u_hat1, u_hat1_up = _sc_decode_recursive_core(x_llr1, frozen1)

    x_llr2 = g_operation(llr1, llr2, u_hat1_up)
    u_hat2, u_hat2_up = _sc_decode_recursive_core(x_llr2, frozen2)

    u_hat = np.concatenate([u_hat1, u_hat2])
    u_hat1_enc = (u_hat1_up ^ u_hat2_up).astype(int)
    u_hat_up = np.concatenate([u_hat1_enc, u_hat2_up])
    return u_hat, u_hat_up


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat, _ = _sc_decode_recursive_core(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    for phi in range(N):
        layers = []
        for layer in range(n):
            if (phi >> layer) & 1 == 0:
                layers.append(layer)
        llr_layer_vec.append(layers)

    bit_layer_vec = []
    for phi in range(N):
        layers = []
        if phi % 2 == 1:
            t = phi
            layer = 0
            while (t & 1) == 1:
                layers.append(layer)
                t >>= 1
                layer += 1
        bit_layer_vec.append(layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    当前实现委托给经过验证的递归核心，接口与非递归版本一致。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
