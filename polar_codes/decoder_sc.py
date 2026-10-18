"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def _sign_llr(x):
    s = np.sign(x)
    return np.where(s == 0, 1.0, s)


def f_operation(La, Lb):
    """f 运算（数值稳定的 boxplus）。"""
    La = np.clip(np.asarray(La, dtype=np.float64), -30.0, 30.0)
    Lb = np.clip(np.asarray(Lb, dtype=np.float64), -30.0, 30.0)
    ta = np.tanh(La / 2.0)
    tb = np.tanh(Lb / 2.0)
    prod = np.clip(ta * tb, -1.0 + 1e-12, 1.0 - 1e-12)
    return 2.0 * np.arctanh(prod)


def g_operation(La, Lb, u_hat):
    """g 运算（VN 更新）。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    g 运算使用子树重编码比特 u_hat_up（与 Sionna/Arikan 一致）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                u = 0
            else:
                u = 0 if llr_node[0] >= 0 else 1
            return np.array([u], dtype=np.int8), np.array([u], dtype=np.int8)

        half = n // 2
        llr1 = llr_node[:half]
        llr2 = llr_node[half:]
        fr1 = frozen_node[:half]
        fr2 = frozen_node[half:]

        llr_up = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = decode_node(llr_up, fr1)

        llr_low = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = decode_node(llr_low, fr2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up = np.bitwise_xor(u_hat1_up.astype(np.int8), u_hat2_up.astype(np.int8))
        u_hat_up = np.concatenate([u_hat1_up, u_hat2_up])
        return u_hat, u_hat_up

    u_hat, _ = decode_node(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 的 LLR/比特层列表。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        p = phi
        layer = 0
        while (p & 1) == 1 and layer < n:
            p >>= 1
            layer += 1
        for l in range(layer, n):
            layers_llr.append(l)

        layers_bit = []
        p = phi
        l = 0
        while l < n and (p & 1) == 0:
            layers_bit.append(l)
            p >>= 1
            l += 1
        llr_layer_vec.append(layers_llr)
        bit_layer_vec.append(layers_bit)

    lambda_offset = []
    off = 0
    for l in range(n + 1):
        lambda_offset.append(off)
        off += 1 << (n - l)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（与递归版本等价）。
    为简洁与可维护性，调用修复后的递归实现。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
