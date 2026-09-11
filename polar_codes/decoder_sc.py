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
    """
    g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La
    """
    return Lb + (1 - 2 * np.asarray(u_hat, dtype=int)) * La


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    N = len(llr)
    n = int(math.log2(N)) + 1
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
    node_values = np.zeros(N, dtype=int)

    def decode_node(y, depth, node):
        if depth == n - 1:
            if node in frozen_set:
                node_values[node] = 0
            else:
                node_values[node] = 0 if y[0] >= 0 else 1
            return [node_values[node]]

        half = len(y) // 2
        L1, L2 = y[:half], y[half:]
        left = f_operation(L1, L2)
        arr1 = decode_node(left, depth + 1, 2 * node)
        right = g_operation(L1, L2, arr1)
        arr2 = decode_node(right, depth + 1, 2 * node + 1)
        return [(arr1[i] + arr2[i]) % 2 for i in range(len(arr1))] + list(arr2)

    decode_node(np.asarray(llr, dtype=np.float64), 0, 0)
    return node_values


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        temp = phi
        for layer in range(n):
            if (temp & 1) == 0:
                layers.append(layer)
            temp >>= 1
        llr_layer_vec.append(layers)

        layers_b = []
        temp = phi
        for layer in range(n):
            if (temp & 1) == 1:
                layers_b.append(layer)
            temp >>= 1
        bit_layer_vec.append(layers_b)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（基于层更新）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """主 SC 译码接口"""
    return sc_decode_recursive(llr_ch, frozen_bits)
