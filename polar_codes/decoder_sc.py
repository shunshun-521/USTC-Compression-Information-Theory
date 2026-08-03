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


def _as_list(llr):
    return np.asarray(llr, dtype=np.float64).tolist()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(math.log2(N)) + 1
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    node_values = np.zeros(N, dtype=int)

    def decode_node(y, depth, node):
        if depth == n - 1:
            if node in frozen_set:
                node_values[node] = 0
            else:
                node_values[node] = 0 if y[0] >= 0 else 1
            return [int(node_values[node])]

        half = len(y) // 2
        l1 = y[:half]
        l2 = y[half:]
        left = f_operation(np.asarray(l1), np.asarray(l2)).tolist()
        arr1 = decode_node(left, depth + 1, 2 * node)
        right = g_operation(np.asarray(l1), np.asarray(l2), np.asarray(arr1)).tolist()
        arr2 = decode_node(right, depth + 1, 2 * node + 1)
        return [(arr1[i] + arr2[i]) % 2 for i in range(len(arr1))] + arr2

    decode_node(_as_list(llr), 0, 0)
    return node_values


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = (1 << layer) - 1

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 0:
                llr_layers.append(layer)
            temp //= 2
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 1:
                bit_layers.append(layer)
            temp //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（显式栈实现，与递归版本等价）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = int(math.log2(len(llr_ch))) + 1
    frozen_set = set(np.where(frozen_bits)[0])
    node_values = np.zeros(len(llr_ch), dtype=int)
    returns = {}

    stack = [('visit', _as_list(llr_ch), 0, 0)]

    while stack:
        kind, *args = stack.pop()
        if kind == 'visit':
            y, depth, node = args
            if depth == n - 1:
                if node in frozen_set:
                    node_values[node] = 0
                else:
                    node_values[node] = 0 if y[0] >= 0 else 1
                returns[(depth, node)] = [int(node_values[node])]
            else:
                half = len(y) // 2
                l1, l2 = y[:half], y[half:]
                left = f_operation(np.asarray(l1), np.asarray(l2)).tolist()
                stack.append(('finish_internal', l1, l2, depth, node))
                stack.append(('visit', left, depth + 1, 2 * node))
        elif kind == 'finish_internal':
            l1, l2, depth, node = args
            arr1 = returns.pop((depth + 1, 2 * node))
            right = g_operation(np.asarray(l1), np.asarray(l2), np.asarray(arr1)).tolist()
            stack.append(('combine', arr1, depth, node))
            stack.append(('visit', right, depth + 1, 2 * node + 1))
        elif kind == 'combine':
            arr1, depth, node = args
            arr2 = returns.pop((depth + 1, 2 * node + 1))
            returns[(depth, node)] = [(arr1[i] + arr2[i]) % 2 for i in range(len(arr1))] + arr2

    return node_values
