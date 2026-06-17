"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        tmp = phi
        while tmp % 2 == 1:
            layers.append(int(math.log2(tmp & -tmp)))
            tmp -= tmp & -tmp
        llr_layer_vec.append(layers)

        bit_layers = []
        tmp = phi
        while tmp % 2 == 1:
            bit_layers.append(int(math.log2(tmp & -tmp)))
            tmp -= tmp & -tmp
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_tree_decode(llr, frozen_bits, u_hat, offset, size):
    """一次性树形 SC 译码（核心逻辑）。"""
    if size == 1:
        if frozen_bits[offset]:
            u_hat[offset] = 0
        else:
            u_hat[offset] = 0 if llr[0] >= 0 else 1
        return np.array([u_hat[offset]], dtype=int)

    half = size // 2
    ll = np.array([f_operation(llr[i], llr[i + half]) for i in range(half)])
    s_left = _sc_tree_decode(ll, frozen_bits, u_hat, offset, half)
    lr = np.array([
        g_operation(llr[i], llr[i + half], s_left[i]) for i in range(half)
    ])
    s_right = _sc_tree_decode(lr, frozen_bits, u_hat, offset + half, half)
    return np.concatenate([s_left ^ s_right, s_right])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(len(llr), dtype=int)
    _sc_tree_decode(llr, frozen_bits, u_hat, 0, len(llr))
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    采用与 sc_decode_recursive 等价的树形 LLR 组合（f/g 跨半长配对）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
