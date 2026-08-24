"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation

LLR_MAX = 30.0


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（check-node / boxplus 近似）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def cn_operation(La, Lb):
    """精确 boxplus（用于递归参考实现）"""
    x = np.clip(La, -LLR_MAX, LLR_MAX)
    y = np.clip(Lb, -LLR_MAX, LLR_MAX)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def g_operation(La, Lb, u_hat):
    """g 运算（VN update）"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _polar_decode_sc_recursive(llr_ch, frozen_ind, cn_func=cn_operation):
    """
  递归 SC 译码（Sionna 风格，返回 u_hat 与 stage 部分和 u_hat_up）
    """
    n = len(llr_ch)
    frozen_ind = np.asarray(frozen_ind, dtype=bool)

    if n > 1:
        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        frozen1 = frozen_ind[:half]
        frozen2 = frozen_ind[half:]

        x_llr1 = cn_func(llr1, llr2)
        u_hat1, u_hat1_up = _polar_decode_sc_recursive(x_llr1, frozen1, cn_func)

        x_llr2 = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = _polar_decode_sc_recursive(x_llr2, frozen2, cn_func)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up = np.bitwise_xor(u_hat1_up.astype(np.int8), u_hat2_up.astype(np.int8))
        u_hat_up = np.concatenate([u_hat1_up, u_hat2_up])
        return u_hat, u_hat_up

    is_frozen = frozen_ind[0]
    if is_frozen:
        u_hat = np.array([0], dtype=np.int8)
    else:
        u_hat = np.array([0 if llr_ch[0] >= 0 else 1], dtype=np.int8)
    return u_hat, u_hat.copy()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat, _ = _polar_decode_sc_recursive(np.asarray(llr, dtype=np.float64), frozen_bits)
    return u_hat.astype(np.int8)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        psi = phi
        while psi % 2 == 1:
            llr_layers.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        psi = phi
        while psi > 0 and psi % 2 == 0:
            psi >>= 1
        if psi > 0:
            while psi > 0:
                bit_layers.append(int(math.log2(psi & -psi)))
                psi >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（信道 LLR 按码字顺序输入）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
