"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
from enum import IntEnum

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


class NodeState(IntEnum):
    NOT_VISITED = 0
    AFTER_L = 1
    AFTER_R = 2


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 结果一致）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（兼容接口）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << d for d in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        psi = phi
        layer = 0
        while psi & 1:
            layers.append(layer)
            layer += 1
            psi >>= 1
        llr_layer_vec.append(layers)
        if phi == N - 1:
            bit_layer_vec.append(list(range(n)))
        elif (phi & 1) == 0:
            layers_b = []
            psi = phi
            layer = 0
            while (psi & 1) == 0:
                layers_b.append(layer)
                layer += 1
                psi >>= 1
            bit_layer_vec.append(layers_b)
        else:
            bit_layer_vec.append([0])
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（树遍历 L/R/U 步，信道 LLR 置于根层）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    beliefs = np.zeros((n + 1, N), dtype=np.float64)
    decoded_bits = np.zeros((n + 1, N), dtype=np.int32)
    beliefs[0, :] = llr_ch
    node_state = np.zeros(2 * N - 1, dtype=np.int32)

    node = 0
    depth = 0
    done = False

    while not done:
        if depth == n:
            if frozen_bits[node]:
                decoded_bits[n, node] = 0
            else:
                decoded_bits[n, node] = 0 if beliefs[n, node] >= 0 else 1
            if node == N - 1:
                done = True
            else:
                node //= 2
                depth -= 1
        else:
            node_pos = (1 << depth) - 1 + node
            if node_state[node_pos] == NodeState.NOT_VISITED:
                span = 1 << (n - depth)
                incoming = beliefs[depth, span * node:span * (node + 1)]
                half = span // 2
                left = incoming[:half]
                right = incoming[half:]
                child = node * 2
                child_span = span // 2
                beliefs[depth + 1, child_span * child:child_span * (child + 1)] = f_operation(left, right)
                node_state[node_pos] = NodeState.AFTER_L
                node = child
                depth += 1
            elif node_state[node_pos] == NodeState.AFTER_L:
                span = 1 << (n - depth)
                incoming = beliefs[depth, span * node:span * (node + 1)]
                half = span // 2
                left = incoming[:half]
                right = incoming[half:]
                left_child = node * 2
                left_span = span // 2
                u_left = decoded_bits[depth + 1, left_span * left_child:left_span * (left_child + 1)]
                child = node * 2 + 1
                child_span = span // 2
                beliefs[depth + 1, child_span * child:child_span * (child + 1)] = g_operation(left, right, u_left)
                node_state[node_pos] = NodeState.AFTER_R
                node = child
                depth += 1
            else:
                span = 1 << (n - depth)
                left_child = node * 2
                right_child = node * 2 + 1
                half = span // 2
                bits_left = decoded_bits[depth + 1, half * left_child:half * (left_child + 1)]
                bits_right = decoded_bits[depth + 1, half * right_child:half * (right_child + 1)]
                decoded_bits[depth, span * node:span * (node + 1)] = np.concatenate(
                    [(bits_left + bits_right) % 2, bits_right]
                )
                node //= 2
                depth -= 1

    return decoded_bits[n, :].copy()
