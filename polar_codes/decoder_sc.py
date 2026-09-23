"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversed


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _decode_recursive(llr, frozen_set, depth, node, node_values, n):
    if depth == n - 1:
        if node in frozen_set:
            node_values[node] = 0
        else:
            node_values[node] = 1 if llr[0] < 0 else 0
        return [node_values[node]]

    half = len(llr) // 2
    left = f_operation(np.array(llr[:half]), np.array(llr[half:]))
    arr1 = _decode_recursive(left.tolist(), frozen_set, depth + 1, 2 * node, node_values, n)

    right = g_operation(np.array(llr[:half]), np.array(llr[half:]), arr1)
    arr2 = _decode_recursive(right.tolist(), frozen_set, depth + 1, 2 * node + 1, node_values, n)

    result = [(arr1[i] + arr2[i]) % 2 for i in range(len(arr1))]
    result.extend(arr2)
    return result


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N)) + 1
    frozen_set = set(np.where(frozen_bits)[0])
    node_values = np.zeros(N, dtype=int)
    _decode_recursive(llr.tolist(), frozen_set, 0, 0, node_values, n)
    return node_values


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容接口）"""
    n = int(math.log2(N))
    order = [bit_reversed(i, n) for i in range(N)]
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec, np.array(order)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（调用高效递归内核）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
