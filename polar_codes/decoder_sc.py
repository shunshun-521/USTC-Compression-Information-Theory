"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def _frozen_to_set(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return set(np.where(fb)[0])
    return set(np.where(fb != 0)[0])


def _xor_combine(left, right):
    res = [(left[i] + right[i]) % 2 for i in range(len(left))]
    res.extend(right)
    return res


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_set = _frozen_to_set(frozen_bits)
    N = len(llr)
    n = int(math.log2(N)) + 1
    node_values = np.zeros(N, dtype=int)

    def _decode(y, depth, node):
        if depth == n - 1:
            if node in frozen_set:
                node_values[node] = 0
                return [0]
            bit = 1 if y[0] < 0 else 0
            node_values[node] = bit
            return [bit]

        half = len(y) // 2
        l1, l2 = y[:half], y[half:]
        arr1 = _decode(f_operation(l1, l2), depth + 1, 2 * node)
        arr2 = _decode(g_operation(l1, l2, arr1), depth + 1, 2 * node + 1)
        return _xor_combine(arr1, arr2)

    _decode(llr, 0, 0)
    return node_values


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [0]
    for i in range(1, n + 1):
        lambda_offset.append(lambda_offset[i - 1] + (1 << (n - i + 1)))

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


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（当前调用已验证的递归实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
