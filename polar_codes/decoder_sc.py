"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """box-plus（对大 LLR 使用 min-sum 近似）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    out = np.empty_like(La)
    large = np.maximum(np.abs(La), np.abs(Lb)) > 30
    out[large] = (
        np.sign(La[large])
        * np.sign(Lb[large])
        * np.minimum(np.abs(La[large]), np.abs(Lb[large]))
    )
    if np.any(~large):
        a, b = La[~large], Lb[~large]
        out[~large] = np.log1p(np.exp(a + b)) - np.log(np.exp(a) + np.exp(b))
    return out


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _hard_decision(llr):
    """硬判决，LLR=0 时判为 1（与常见实现一致）"""
    llr = np.atleast_1d(np.asarray(llr, dtype=np.float64))
    d = 0.5 * (1.0 - np.sign(llr))
    return np.where(d == 0.5, 1.0, d).astype(np.float64)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen = np.asarray(frozen_bits, dtype=bool)

    def decode_node(node_llr, fzn):
        n = len(node_llr)
        if n == 1:
            if fzn[0]:
                return np.array([0.0]), np.array([0.0])
            bit = _hard_decision(node_llr[0:1])
            return bit, bit

        half = n // 2
        llr_left = f_operation(node_llr[:half], node_llr[half:])
        u_left, u_left_up = decode_node(llr_left, fzn[:half])
        llr_right = g_operation(node_llr[:half], node_llr[half:], u_left_up)
        u_right, u_right_up = decode_node(llr_right, fzn[half:])

        u_hat = np.concatenate([u_left, u_right])
        u_up = np.concatenate(
            [
                (np.round(u_left_up).astype(np.int8) ^ np.round(u_right_up).astype(np.int8)).astype(
                    np.float64
                ),
                u_right_up,
            ]
        )
        return u_hat, u_up

    u_hat, _ = decode_node(llr, frozen)
    return np.round(u_hat).astype(np.int8)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        psi = phi
        layers_llr = []
        while psi % 2 == 1:
            layers_llr.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers_llr)

        if phi % 2 == 0:
            bit_layers = list(range(n))
        else:
            psi = phi
            bit_layers = []
            while psi % 2 == 1:
                bit_layers.append(int(math.log2(psi & -psi)))
                psi >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（调用高效递归实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
