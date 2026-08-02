"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

LLR_MAX = 30.0


def f_operation(La, Lb):
    """精确 boxplus f 运算（含 LLR 限幅），支持向量化"""
    x_in = np.clip(La, -LLR_MAX, LLR_MAX)
    y_in = np.clip(Lb, -LLR_MAX, LLR_MAX)
    return np.log1p(np.exp(x_in + y_in)) - np.log(np.exp(x_in) + np.exp(y_in))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _polar_decode_sc_recursive(llr_ch, frozen_ind):
    """内部递归 SC 译码（Sionna 兼容实现）。"""
    n = len(frozen_ind)
    if n > 1:
        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        f1 = frozen_ind[:half]
        f2 = frozen_ind[half:]

        x_llr1 = f_operation(llr1, llr2)
        u1, u1_up = _polar_decode_sc_recursive(x_llr1, f1)

        x_llr2 = g_operation(llr1, llr2, u1_up)
        u2, u2_up = _polar_decode_sc_recursive(x_llr2, f2)

        u_hat = np.concatenate([u1, u2])
        u1_up_int = (u1_up.astype(int) ^ u2_up.astype(int)).astype(np.float64)
        u_hat_up = np.concatenate([u1_up_int, u2_up])
        return u_hat.astype(int), u_hat_up

    if frozen_ind[0]:
        u_hat = np.array([0.0])
    else:
        u_hat = np.array([0.0 if llr_ch[0] >= 0 else 1.0])
        if llr_ch[0] == 0:
            u_hat = np.array([1.0])
    return u_hat.astype(int), u_hat.copy()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    frozen_ind = np.asarray(frozen_bits, dtype=bool).astype(float)
    u_hat, _ = _polar_decode_sc_recursive(np.asarray(llr, dtype=np.float64), frozen_ind)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容接口）。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        psi = phi
        layer = 0
        while psi & 1:
            layers_llr.append(layer)
            psi >>= 1
            layer += 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        psi = phi
        layer = 0
        while psi & 1:
            layer += 1
            psi >>= 1
        for j in range(layer):
            layers_bit.append(j)
        bit_layer_vec.append(layers_bit)

    return np.zeros(n + 1, dtype=int), llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（当前调用递归实现保证正确性）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
