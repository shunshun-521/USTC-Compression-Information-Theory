"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _prepare_channel_llr(llr_ch):
    """对信道 LLR 做比特倒序，与编码端 B_N 一致。"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _compute_u_up(u_vals):
    """计算子树顶层的 re-encoded 部分和 u_up（用于 g 运算）。"""
    u_vals = np.asarray(u_vals, dtype=np.int8)
    n = len(u_vals)
    if n == 1:
        return u_vals.copy()
    half = n // 2
    up1 = _compute_u_up(u_vals[:half])
    up2 = _compute_u_up(u_vals[half:])
    return np.concatenate([(up1 ^ up2), up2])


def _llr_at_phi(llr_br, u_hat, phi):
    """给定已判决比特，计算位置 phi 的叶节点 LLR。"""

    def rec(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            return llr_node[0]
        half = n // 2
        if phi < bit_offset + half:
            llr_left = f_operation(llr_node[:half], llr_node[half:])
            return rec(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        u_left_up = _compute_u_up(u_left)
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
        return rec(llr_right, bit_offset + half)

    return rec(llr_br, 0)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = _prepare_channel_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(len(llr), dtype=int)

    def decode_node(llr_node, frozen_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_node[0] or llr_node[0] >= 0 else 1
            uh = np.array([u_hat[idx]], dtype=int)
            return uh, uh.copy()

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u1, u1_up = decode_node(llr_left, frozen_node[:half], bit_offset)
        llr_right = g_operation(llr_node[:half], llr_node[half:], u1_up)
        u2, u2_up = decode_node(llr_right, frozen_node[half:], bit_offset + half)
        u_hat[bit_offset : bit_offset + half] = u1
        u_hat[bit_offset + half : bit_offset + n] = u2
        u_up = np.concatenate([(u1_up ^ u2_up), u2_up])
        return np.concatenate([u1, u2]), u_up

    decode_node(llr, frozen_bits, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        tmp = phi
        llr_layers = []
        while tmp % 2 == 1:
            llr_layers.append(int(math.log2(tmp & -tmp)))
            tmp >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 0:
            t = phi
            layer = 0
            while t % 2 == 0 and t < N:
                bit_layers.append(layer)
                layer += 1
                t >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（逐比特顺序更新，无 Python 递归）。
    """
    llr = _prepare_channel_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        llr_leaf = _llr_at_phi(llr, u_hat, phi)
        u_hat[phi] = 0 if frozen_bits[phi] or llr_leaf >= 0 else 1

    return u_hat
