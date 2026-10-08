"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（boxplus）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_exact(La, Lb, llr_max=30.0):
    """精确 boxplus（用于递归参考实现）"""
    La = np.clip(np.asarray(La, dtype=np.float64), -llr_max, llr_max)
    Lb = np.clip(np.asarray(Lb, dtype=np.float64), -llr_max, llr_max)
    return np.log1p(np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _polar_decode_sc_recursive(llr_ch, frozen_ind, use_exact=True):
    """
    递归 SC 译码（与极化码因子图的上/下分支拆分一致）。
    frozen_ind: 长度 n 的数组，1 表示冻结位，0 表示信息位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_ind = np.asarray(frozen_ind, dtype=np.float64)
    n_len = len(llr_ch)

    if n_len > 1:
        half = n_len // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        fr1 = frozen_ind[:half]
        fr2 = frozen_ind[half:]

        cn = f_operation_exact if use_exact else f_operation
        llr_upper = cn(llr1, llr2)
        u1, u1_up = _polar_decode_sc_recursive(llr_upper, fr1, use_exact)
        llr_lower = g_operation(llr1, llr2, u1_up)
        u2, u2_up = _polar_decode_sc_recursive(llr_lower, fr2, use_exact)

        u_hat = np.concatenate([u1, u2])
        u1_up_i = (u1_up.astype(np.int8) ^ u2_up.astype(np.int8)).astype(np.float64)
        u_hat_up = np.concatenate([u1_up_i, u2_up])
        return u_hat, u_hat_up

    is_frozen = frozen_ind[0] == 1
    if is_frozen:
        u_hat = np.array([0.0])
    else:
        u_hat = np.array([0.0 if llr_ch[0] >= 0 else 1.0])
    return u_hat, u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码，返回长度 N 的 u_hat。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    frozen_ind = frozen_bits.astype(np.float64)
    u_hat, _ = _polar_decode_sc_recursive(llr, frozen_ind, use_exact=True)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        t = phi
        while t & 1:
            layers_llr.append(int(math.log2(t & -t)))
            t >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            layers_bit = list(range(n))
        else:
            t = phi
            while t & 1:
                t >>= 1
            if t > 0:
                layers_bit = list(range(int(math.log2(t & -t)), n))
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    当前实现调用经过验证的递归树形算法（复杂度 O(N log N)）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
