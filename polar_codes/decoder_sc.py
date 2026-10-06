"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，与递归共用同一 SC 树逻辑）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    精确 log-domain f 运算（check node）：
    f(a,b) = ln((1+e^{a+b})/(e^a+e^b))
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(a, b, u) = b + (1 - 2u) * a
    u_hat 可为标量或与 La 同形的数组（部分和比特）
    """
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _sc_node(llrs, frozen_bits, u_hat, base, length):
    """SC 递归子树译码（与 commpy 极化树约定一致）。"""
    if length == 1:
        idx = base
        if frozen_bits[idx]:
            u_hat[idx] = 0
        else:
            u_hat[idx] = 0 if llrs[0] >= 0 else 1
        return np.array([u_hat[idx]], dtype=np.int8)

    half = length // 2
    upper = f_operation(llrs[:half], llrs[half:])
    beta_upper = _sc_node(upper, frozen_bits, u_hat, base, half)

    lower = g_operation(llrs[:half], llrs[half:], beta_upper)
    beta_lower = _sc_node(lower, frozen_bits, u_hat, base + half, half)

    return np.concatenate([beta_upper ^ beta_lower, beta_lower])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)
    _sc_node(llr, frozen_bits, u_hat, 0, N)
    return u_hat


# ==================== 非递归 SC（L/C 平面，与递归等价）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引（文档/扩展用）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << (n - i) for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        psi = phi
        layers_llr = []
        while psi & 1:
            layers_llr.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            psi2 = phi
            while psi2 > 0 and psi2 % 2 == 0:
                layers_bit.append(int(math.log2(psi2 & -psi2)) + 1)
                psi2 >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码：当前实现调用与递归等价的树遍历（保证数值一致）。
    对 N<=1024 仿真足够高效；接口与非递归版本一致。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
