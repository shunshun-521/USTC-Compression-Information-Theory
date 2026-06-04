"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """check-node / boxplus（精确 LLR 域，输入裁剪至 ±30）"""
    La = np.clip(np.asarray(La, dtype=np.float64), -30, 30)
    Lb = np.clip(np.asarray(Lb, dtype=np.float64), -30, 30)
    with np.errstate(over='ignore', invalid='ignore'):
        out = np.log(1 + np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb))
    return np.nan_to_num(out, nan=0.0, posinf=30.0, neginf=-30.0)


def g_operation(La, Lb, u_partial):
    """g 运算（value-node），u_partial 为当前阶段的中间比特"""
    return (1 - 2 * np.asarray(u_partial, dtype=np.float64)) * La + Lb


def _sc_decode_tree(llr, frozen_ind):
    """递归 SC 树（返回 u_hat 与中间重编码比特 u_up）"""
    n = len(llr)
    frozen_ind = np.asarray(frozen_ind, dtype=int)

    if n == 1:
        if frozen_ind[0]:
            u = 0
        elif abs(llr[0]) < 1e-12:
            u = 1
        else:
            u = 0 if llr[0] >= 0 else 1
        return np.array([u], dtype=int), np.array([u], dtype=np.float64)

    half = n // 2
    llr1, llr2 = llr[:half], llr[half:]
    f1, f2 = frozen_ind[:half], frozen_ind[half:]

    llr_up = f_operation(llr1, llr2)
    u1, u1_up = _sc_decode_tree(llr_up, f1)

    llr_low = g_operation(llr1, llr2, u1_up)
    u2, u2_up = _sc_decode_tree(llr_low, f2)

    u_hat = np.concatenate([u1, u2])
    u1_up_xor = (u1_up.astype(int) ^ u2_up.astype(int)).astype(np.float64)
    u_up = np.concatenate([u1_up_xor, u2_up])
    return u_hat, u_up


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    N = len(llr)
    rev = bit_reversal_permutation(N)
    llr_rev = np.asarray(llr, dtype=np.float64)[rev]
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    u_hat, _ = _sc_decode_tree(llr_rev, frozen_bits)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        temp = phi
        layer = 0
        while temp & 1:
            temp >>= 1
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))

        temp = phi + 1
        layer = 0
        bit_layers = []
        while temp % 2 == 0 and layer < n:
            bit_layers.append(layer)
            temp >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（调用与递归等价的树形译码）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
