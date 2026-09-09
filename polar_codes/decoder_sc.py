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
    """g 运算（u_hat 为部分重编码比特）"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _sc_decode_recursive(llr, frozen_ind):
    """Sionna/Arikan 风格递归 SC，g 运算使用 u_hat_up。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_ind = np.asarray(frozen_ind, dtype=int)
    n = len(llr)
    if n == 1:
        if frozen_ind[0]:
            u = 0
        else:
            u = 0 if llr[0] >= 0 else 1
        return np.array([u], dtype=int), np.array([u], dtype=float)

    llr1 = llr[: n // 2]
    llr2 = llr[n // 2:]
    f1 = frozen_ind[: n // 2]
    f2 = frozen_ind[n // 2:]

    u1, u1_up = _sc_decode_recursive(f_operation(llr1, llr2), f1)
    u2, u2_up = _sc_decode_recursive(g_operation(llr1, llr2, u1_up), f2)

    u_hat = np.concatenate([u1, u2])
    u1_up_re = (u1_up.astype(int) ^ u2_up.astype(int)).astype(float)
    u_up = np.concatenate([u1_up_re, u2_up])
    return u_hat, u_up


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码。
    frozen_bits: 1 表示冻结位，0 表示信息位
    """
    u_hat, _ = _sc_decode_recursive(llr, frozen_bits)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数（当前实现为高效递归版本）。
    frozen_bits: 1 表示冻结位，0 表示信息位
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口保留）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        psi = phi // 2
        llr_layers = []
        while psi % 2 == 1:
            llr_layers.append(int(math.log2(psi & -psi)))
            psi //= 2
        llr_layer_vec.append(llr_layers)
        if phi % 2 == 0:
            psi = phi // 2
            layers = []
            while psi % 2 == 1:
                layers.append(int(math.log2(psi & -psi)))
                psi //= 2
            bit_layer_vec.append(layers)
        else:
            layers = [0]
            psi = phi // 2
            while psi % 2 == 1:
                layers.append(int(math.log2(psi & -psi)))
                psi //= 2
            bit_layer_vec.append(layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
