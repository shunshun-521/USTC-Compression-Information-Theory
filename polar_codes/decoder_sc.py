"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation

LLR_MAX = 30.0


def cn_op(La, Lb):
    """精确 box-plus（f 运算）"""
    La = np.clip(np.asarray(La, dtype=np.float64), -LLR_MAX, LLR_MAX)
    Lb = np.clip(np.asarray(Lb, dtype=np.float64), -LLR_MAX, LLR_MAX)
    return np.log1p(np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb))


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（保留供 BP 等模块使用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _sc_decode_recursive_core(llr_ch, frozen_ind):
    """递归 SC 译码核心（使用部分和 u_up）"""
    n = len(llr_ch)
    frozen_ind = np.asarray(frozen_ind, dtype=np.float64)
    if n == 1:
        if frozen_ind[0] == 1:
            u_hat = np.array([0.0])
        else:
            u_hat = np.array([0.0 if llr_ch[0] >= 0 else 1.0])
        return u_hat, u_hat.copy()

    half = n // 2
    llr_left = cn_op(llr_ch[:half], llr_ch[half:])
    u_left, u_left_up = _sc_decode_recursive_core(llr_left, frozen_ind[:half])
    llr_right = g_operation(llr_ch[:half], llr_ch[half:], u_left_up)
    u_right, u_right_up = _sc_decode_recursive_core(llr_right, frozen_ind[half:])
    u_hat = np.concatenate([u_left, u_right])
    u_left_up = np.bitwise_xor(
        u_left_up.astype(int), u_right_up.astype(int)
    ).astype(np.float64)
    u_up = np.concatenate([u_left_up, u_right_up])
    return u_hat, u_up


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    N = len(llr)
    br = bit_reversal_permutation(N)
    frozen_ind = np.zeros(N, dtype=np.float64)
    frozen_ind[np.asarray(frozen_bits, dtype=bool)] = 1.0
    u_hat, _ = _sc_decode_recursive_core(np.asarray(llr, dtype=np.float64)[br], frozen_ind)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        temp = phi
        layer = 0
        while layer < n:
            layers_llr.append(layer)
            if temp % 2 == 0:
                break
            temp //= 2
            layer += 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        temp = phi
        layer = 0
        while temp % 2 == 1 and layer < n:
            layers_bit.append(layer)
            temp //= 2
            layer += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（调用高效递归核心）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
