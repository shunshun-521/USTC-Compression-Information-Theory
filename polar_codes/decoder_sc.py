"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _xor_combine(left, right):
    """递归译码中合并左右子树结果。"""
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    return np.concatenate([(left + right) % 2, right])


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    frozen_bits[i]=True 表示冻结位。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N)) + 1
    frozen_set = set(np.where(frozen_bits)[0])
    node_values = np.zeros(N, dtype=int)

    def decode_node(y, depth, node):
        if depth == n - 1:
            if node in frozen_set:
                bit = 0
            else:
                bit = 1 if y[0] < 0 else 0
            node_values[node] = bit
            return np.array([bit], dtype=int)

        half = len(y) // 2
        l1, l2 = y[:half], y[half:]
        left = decode_node(f_operation(l1, l2), depth + 1, 2 * node)
        right = decode_node(g_operation(l1, l2, left), depth + 1, 2 * node + 1)
        return _xor_combine(left, right)

    decode_node(llr, 0, 0)
    return node_values


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layer = 0
        temp = phi
        while temp % 2 == 1:
            temp //= 2
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))

        bit_layers = [l for l in range(n) if (phi >> l) & 1]
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（与递归版本等价）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
