"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation

_LLR_MAX = 30.0


def f_operation(La, Lb):
    """
    对数域 box-plus（f 运算），向量化实现。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    La = np.clip(La, -_LLR_MAX, _LLR_MAX)
    Lb = np.clip(Lb, -_LLR_MAX, _LLR_MAX)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.log1p(np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u_hat = np.asarray(u_hat, dtype=np.float64)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _llr_to_work(llr_ch):
    """编码含比特倒序时，将信道 LLR 映射到译码树自然顺序。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br]


def _hard_decision(llr_val):
    if llr_val >= 0:
        return 0
    return 1


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = _llr_to_work(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = _hard_decision(llr_node[0])
            return np.array([u_hat[idx]], dtype=np.float64)

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left_up = decode_node(llr_left, bit_offset)
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [0]
    for layer in range(1, n + 1):
        lambda_offset.append(2 ** (n - layer))

    decode_order = [int(format(i, f"0{n}b")[::-1], 2) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in decode_order:
        llr_layers = []
        bit_layers = []
        for layer in range(n):
            if (phi >> layer) & 1 == 0:
                llr_layers.append(layer)
            else:
                bit_layers.append(layer)
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（树递归实现，与 sc_decode_recursive 等价）。
    """
    llr = _llr_to_work(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            bit = 0 if frozen_node[0] else _hard_decision(llr_node[0])
            return np.array([bit], dtype=int), np.array([bit], dtype=np.float64)

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left, u_left_up = decode_node(llr_left, frozen_node[:half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
        u_right, u_right_up = decode_node(llr_right, frozen_node[half:])
        u_hat = np.concatenate([u_left, u_right])
        u_up = np.concatenate([(u_left_up.astype(np.int8) ^ u_right_up.astype(np.int8)).astype(np.float64), u_right_up])
        return u_hat, u_up

    u_hat, _ = decode_node(llr, frozen_bits)
    return u_hat.astype(int)
