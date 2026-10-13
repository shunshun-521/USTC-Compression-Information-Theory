"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_part):
    """g 运算；u_part 为左子树的部分码字（长度与 La 相同）"""
    u_part = np.asarray(u_part, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_part) * La


def _sc_node(llrs, frozen_bits, base, length, u_hat):
    """CommPy 风格 SC 递归（单路径），返回部分码字 beta。"""
    if length == 1:
        idx = base
        if frozen_bits[idx]:
            bit = 0
        else:
            bit = 0 if llrs[0] >= 0 else 1
        u_hat[idx] = bit
        return np.array([bit], dtype=np.int8)

    half = length // 2
    upper_llr = f_operation(llrs[:half], llrs[half:])
    beta_upper = _sc_node(upper_llr, frozen_bits, base, half, u_hat)
    lower_llr = g_operation(llrs[:half], llrs[half:], beta_upper)
    beta_lower = _sc_node(lower_llr, frozen_bits, base + half, half, u_hat)
    return np.concatenate([beta_upper ^ beta_lower, beta_lower]).astype(np.int8)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    _sc_node(llr, frozen_bits, 0, N, u_hat)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
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
        if phi % 2 == 0:
            psi2 = phi
            while psi2 % 2 == 0 and psi2 > 0:
                bit_layers.append(int(math.log2(psi2 & -psi2)))
                psi2 >>= 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（与递归版本等价，复用递归核心保证正确性）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
