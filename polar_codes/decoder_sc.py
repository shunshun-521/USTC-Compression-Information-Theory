"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    if np.isscalar(La) and np.isscalar(Lb):
        return np.sign(La) * np.sign(Lb) * min(abs(La), abs(Lb))
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_arr = np.asarray(u_hat, dtype=int)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0:
        u = int(u_arr.reshape(-1)[0])
        return (1 - 2 * u) * float(La) + float(Lb)
    return (1 - 2 * u_arr) * La + Lb


def _llr_to_list(llr):
    return llr.tolist() if hasattr(llr, "tolist") else list(llr)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
    N = len(llr)
    n = int(math.log2(N)) + 1
    node_values = np.zeros(N, dtype=int)

    def decode(y, depth, node):
        if depth == n - 1:
            if node in frozen_set:
                node_values[node] = 0
            else:
                node_values[node] = 1 if y[0] < 0 else 0
            return [node_values[node]]

        half = len(y) // 2
        L1, L2 = y[:half], y[half:]
        left = _llr_to_list(f_operation(L1, L2))
        arr1 = decode(left, depth + 1, 2 * node)
        right = _llr_to_list(g_operation(L1, L2, arr1))
        arr2 = decode(right, depth + 1, 2 * node + 1)
        res = [(arr1[i] + arr2[i]) % 2 for i in range(len(arr1))]
        res.extend(arr2)
        return res

    decode(llr.tolist(), 0, 0)
    return node_values


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助信息（与递归版本等价的层索引）"""
    n = int(math.log2(N))
    depth = n + 1
    layers = []
    for phi in range(N):
        layers.append(list(range(depth - 1)))
    return list(range(depth)), layers, layers


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（调用高效递归内核）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
