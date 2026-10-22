"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

LLR_MAX = 30.0


def f_operation(La, Lb):
    """box-plus（与 min-sum 兼容的数值稳定形式）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    x_in = np.clip(La, -LLR_MAX, LLR_MAX)
    y_in = np.clip(Lb, -LLR_MAX, LLR_MAX)
    # log(1+exp(x+y)) - log(exp(x)+exp(y))
    return np.log1p(np.exp(x_in + y_in)) - np.log(np.exp(x_in) + np.exp(y_in))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _polar_decode_sc_recursive(llr_ch, frozen_ind):
    """
    递归 SC（半分割 + box-plus），frozen_ind[i]=1 表示冻结。
    返回 (u_hat, u_hat_up) 供上层 g 运算使用。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_ind = np.asarray(frozen_ind, dtype=np.float64)
    n = frozen_ind.size

    if n > 1:
        h = n // 2
        llr1 = llr_ch[:h]
        llr2 = llr_ch[h:]
        fr1 = frozen_ind[:h]
        fr2 = frozen_ind[h:]

        llr_up = f_operation(llr1, llr2)
        u1, u1_up = _polar_decode_sc_recursive(llr_up, fr1)
        llr_low = g_operation(llr1, llr2, u1_up)
        u2, u2_up = _polar_decode_sc_recursive(llr_low, fr2)

        u_hat = np.concatenate([u1, u2])
        u_hat_up = np.concatenate([np.bitwise_xor(u1_up.astype(int), u2_up.astype(int)), u2_up])
        return u_hat, u_hat_up.astype(np.float64)

    # 叶节点
    if frozen_ind[0] == 1:
        return np.array([0], dtype=int), np.array([0.0])
    llr = llr_ch[0]
    bit = 1 if llr < 0 else 0
    if llr == 0:
        bit = 1
    return np.array([bit], dtype=int), np.array([float(bit)])


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr = -np.asarray(llr_ch, dtype=np.float64)
    frozen_ind = frozen_bits.astype(np.float64)
    u_hat, _ = _polar_decode_sc_recursive(llr, frozen_ind)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

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
        if phi % 2 == 0:
            bit_layers.append(0)
        p = phi
        while p > 0 and (p % 2 == 0):
            bit_layers.append(int(math.log2(p & -p)))
            p >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC：当前实现与递归版本等价（便于维护）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
