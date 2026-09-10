"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_exact(La, Lb):
    """精确 log-domain f 运算（box-plus）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _decode_node(llr, frozen_bits, base, length, u_hat, use_min_sum=True):
    """递归译码子树，返回部分和 beta。"""
    f_fn = f_operation if use_min_sum else f_operation_exact

    if length == 1:
        idx = base
        if frozen_bits[idx]:
            u_hat[idx] = 0
        else:
            u_hat[idx] = 0 if llr[0] >= 0 else 1
        return np.array([u_hat[idx]], dtype=int)

    half = length // 2
    upper = f_fn(llr[:half], llr[half:])
    beta_upper = _decode_node(upper, frozen_bits, base, half, u_hat, use_min_sum)
    lower = g_operation(llr[:half], llr[half:], beta_upper)
    beta_lower = _decode_node(lower, frozen_bits, base + half, half, u_hat, use_min_sum)
    return np.concatenate([beta_upper ^ beta_lower, beta_lower])


def sc_decode_recursive(llr, frozen_bits, use_min_sum=True):
    """
    递归 SC 译码。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    _decode_node(llr, frozen_bits, 0, N, u_hat, use_min_sum)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（保留接口）。"""
    import math
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        temp = phi
        for layer in range(n):
            if (temp & 1) == 0:
                llr_layers.append(layer)
            temp >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        temp = phi
        for layer in range(n):
            if (temp & 1) == 1:
                bit_layers.append(layer)
            temp >>= 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits, use_min_sum=True):
    """非递归 SC 译码（当前使用递归实现）。"""
    return sc_decode_recursive(llr_ch, frozen_bits, use_min_sum=use_min_sum)
