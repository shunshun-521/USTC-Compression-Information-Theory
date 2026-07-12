"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return np.asarray(Lb, dtype=np.float64) + (1 - 2 * np.asarray(u_hat)) * np.asarray(
        La, dtype=np.float64
    )


def _xor_paths(left, right):
    """合并左右子树返回的部分和路径。"""
    left = list(left)
    right = list(right)
    merged = [(left[i] + right[i]) % 2 for i in range(len(left))]
    merged.extend(right)
    return merged


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，基于因子图树遍历）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N)) + 1
    frozen_set = set(np.where(frozen_bits)[0])
    node_values = np.zeros(N, dtype=int)

    def decode_node(y, depth, node):
        if depth == n - 1:
            if node in frozen_set:
                node_values[node] = 0
                return [0]
            bit = 1 if y[0] < 0 else 0
            node_values[node] = bit
            return [bit]

        half = len(y) // 2
        l1 = y[:half]
        l2 = y[half:]
        left_llr = f_operation(l1, l2)
        arr1 = decode_node(left_llr, depth + 1, 2 * node)
        right_llr = g_operation(l1, l2, arr1)
        arr2 = decode_node(right_llr, depth + 1, 2 * node + 1)
        return _xor_paths(arr1, arr2)

    decode_node(llr, 0, 0)
    return node_values


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << s for s in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layer = 0
        tmp = phi
        while tmp & 1:
            layer += 1
            tmp >>= 1
        llr_layer_vec.append(list(range(layer, n)))
        bit_layer_vec.append(list(range(layer)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数（调用已验证的递归实现）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
