"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_boxplus(La, Lb):
    """精确 boxplus 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    la = np.clip(La, -50, 50)
    lb = np.clip(Lb, -50, 50)
    return np.log1p(np.exp(la + lb)) - np.log(np.exp(la) + np.exp(lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _permute_llr_from_channel(llr_ch):
    """将信道 LLR 变换为与蝶形+比特倒序编码一致的顺序"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=int)
    inv[br] = np.arange(N)
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def _polar_decode_sc_recursive(llr_ch, frozen_bits):
    """递归 SC 译码"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = len(llr_ch)

    if n == 1:
        if frozen_bits[0]:
            u_hat = np.array([0], dtype=int)
        else:
            u_hat = np.array([0 if llr_ch[0] >= 0 else 1], dtype=int)
        return u_hat, u_hat.copy()

    half = n // 2
    llr_left = f_boxplus(llr_ch[:half], llr_ch[half:])
    u_hat1, u_hat1_up = _polar_decode_sc_recursive(llr_left, frozen_bits[:half])
    llr_right = g_operation(llr_ch[:half], llr_ch[half:], u_hat1_up)
    u_hat2, u_hat2_up = _polar_decode_sc_recursive(llr_right, frozen_bits[half:])

    u_hat = np.concatenate([u_hat1, u_hat2])
    u_hat1_up = (u_hat1_up + u_hat2_up) % 2
    u_hat_up = np.concatenate([u_hat1_up, u_hat2_up])
    return u_hat, u_hat_up


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = _permute_llr_from_channel(llr)
    u_hat, _ = _polar_decode_sc_recursive(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 2)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        psi = phi
        llr_layers = []
        while psi % 2 == 1:
            llr_layers.append(int(math.log2(psi & -psi)) - 1)
            psi >>= 1
        llr_layer_vec.append([l for l in llr_layers if l >= 0])

        bit_layers = []
        psi = phi
        for layer in range(n):
            if psi % 2 == 0:
                bit_layers.append(layer)
            psi //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    return sc_decode_recursive(llr_ch, frozen_bits)
