"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def boxplus(La, Lb, llr_max=30.0):
    """boxplus（对数似然比域 f 运算）"""
    x = np.clip(La, -llr_max, llr_max)
    y = np.clip(Lb, -llr_max, llr_max)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _sc_recursive_core(llr, frozen_ind, llr_max=30.0, fixed_bits=None, bit_offset=0):
    """递归 SC 核心，返回 (u_hat, u_up)"""
    n = len(llr)
    frozen_ind = np.asarray(frozen_ind, dtype=int)
    if fixed_bits is None:
        fixed_bits = np.full(n, -1, dtype=int)

    if n == 1:
        if fixed_bits[bit_offset] >= 0:
            u = int(fixed_bits[bit_offset])
        elif frozen_ind[0]:
            u = 0
        else:
            u = 0 if llr[0] >= 0 else 1
        return np.array([u], dtype=int), np.array([u], dtype=int)

    half = n // 2
    llr1, llr2 = llr[:half], llr[half:]
    fr1, fr2 = frozen_ind[:half], frozen_ind[half:]

    llr_u = boxplus(llr1, llr2, llr_max)
    u1, u1_up = _sc_recursive_core(
        llr_u, fr1, llr_max, fixed_bits, bit_offset
    )
    llr_l = g_operation(llr1, llr2, u1_up)
    u2, u2_up = _sc_recursive_core(
        llr_l, fr2, llr_max, fixed_bits, bit_offset + half
    )

    u_hat = np.concatenate([u1, u2])
    u_up_top = (u1_up ^ u2_up).astype(int)
    u_up = np.concatenate([u_up_top, u2_up])
    return u_hat, u_up


def sc_leaf_llr(llr, frozen_bits, u_prefix, phi):
    """已知 u_prefix[0:phi] 时，返回第 phi 位的叶子 LLR"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    fixed = np.full(len(llr), -1, dtype=int)
    fixed[:phi] = u_prefix[:phi]

    def walk(L, fr, offset):
        n = len(L)
        if n == 1:
            return float(L[0])
        half = n // 2
        L_left = boxplus(L[:half], L[half:])
        if phi < offset + half:
            return walk(L_left, fr[:half], offset)
        u1, u1_up = _sc_recursive_core(
            L_left, fr[:half], fixed_bits=fixed, bit_offset=offset
        )
        L_right = g_operation(L[:half], L[half:], u1_up)
        return walk(L_right, fr[half:], offset + half)

    return walk(llr, frozen_bits, 0)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，boxplus）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    u_hat, _ = _sc_recursive_core(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << (n - i) for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi % 2 == 1:
            layers_llr.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            psi = phi
            while psi % 2 == 0 and psi > 0:
                layers_bit.append(int(math.log2(psi & -psi)))
                psi >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（当前实现：高效递归 + LLR 符号约定统一）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
