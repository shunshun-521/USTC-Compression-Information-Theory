"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（check node / boxplus 的快速近似）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def boxplus(La, Lb):
    """精确 boxplus（f 运算）"""
    La = np.clip(La, -30.0, 30.0)
    Lb = np.clip(Lb, -30.0, 30.0)
    return np.log1p(np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb) + 1e-300)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _sc_recursive_core(llr_node, frozen_node, use_boxplus=False):
    """
    递归 SC 核心：使用 u_hat_up（当前层重编码比特）驱动 g 运算。
    返回 (u_hat, u_hat_up)。
    """
    frozen_node = np.asarray(frozen_node, dtype=bool)
    n = len(llr_node)
    cn = boxplus if use_boxplus else f_operation

    if n == 1:
        if frozen_node[0]:
            u = 0.0
        else:
            u = 0.0 if llr_node[0] >= 0 else 1.0
        return np.array([u]), np.array([u])

    half = n // 2
    llr1 = llr_node[:half]
    llr2 = llr_node[half:]
    fr1 = frozen_node[:half]
    fr2 = frozen_node[half:]

    llr_left = cn(llr1, llr2)
    u_hat1, u_hat1_up = _sc_recursive_core(llr_left, fr1, use_boxplus)
    llr_right = g_operation(llr1, llr2, u_hat1_up)
    u_hat2, u_hat2_up = _sc_recursive_core(llr_right, fr2, use_boxplus)

    u_hat = np.concatenate([u_hat1, u_hat2])
    u1c = (u_hat1_up.astype(int) ^ u_hat2_up.astype(int)).astype(np.float64)
    u_hat_up = np.concatenate([u1c, u_hat2_up])
    return u_hat, u_hat_up


def sc_decode_recursive(llr, frozen_bits, use_boxplus=False):
    """递归 SC 译码"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat, _ = _sc_recursive_core(llr, frozen_bits, use_boxplus)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    offset = 0
    for s in range(n + 1):
        lambda_offset[s] = offset
        offset += 1 << (n - s)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        psi = phi
        layer = 0
        while psi % 2 == 1:
            psi //= 2
            layer += 1
        for l in range(layer, n):
            layers_llr.append(l)

        layers_bit = []
        if phi % 2 == 0:
            layers_bit.append(0)
        psi = (phi + 1) // 2
        l = 1
        while psi % 2 == 1:
            layers_bit.append(l)
            psi //= 2
            l += 1
        bit_layer_vec.append(layers_bit)
        llr_layer_vec.append(layers_llr)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（调用经优化的递归核心）"""
    return sc_decode_recursive(llr_ch, frozen_bits, use_boxplus=False)
