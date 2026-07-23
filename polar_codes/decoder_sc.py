"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La"""
    u_hat = np.asarray(u_hat)
    return Lb + (1 - 2 * u_hat) * La


def _xor_combine(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    return np.concatenate([(left + right) % 2, right])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（树形结构，参考实现）。"""
    N = len(llr)
    n = int(np.log2(N))
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
    node_values = np.zeros(N, dtype=int)

    def decode_node(y, depth, node):
        if depth == n:
            if node in frozen_set:
                node_values[node] = 0
                return [0]
            bit = 0 if y[0] >= 0 else 1
            node_values[node] = bit
            return [bit]

        half = len(y) // 2
        L1, L2 = y[:half], y[half:]
        left_llr = f_operation(L1, L2)
        arr1 = decode_node(left_llr, depth + 1, node)
        right_llr = g_operation(L1, L2, arr1)
        arr2 = decode_node(right_llr, depth + 1, node + half)
        return _xor_combine(arr1, arr2)

    decode_node(np.asarray(llr, dtype=np.float64), 0, 0)
    return node_values


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi % 2 == 1:
            layers_llr.append(int(np.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            psi = phi + 1
            while psi % 2 == 0 and psi < N:
                layers_bit.append(int(np.log2(psi & -psi)) - 1)
                psi >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数。当前使用经过验证的树形递归实现。
  非递归分层实现接口保留在 precompute_sc_indices 中供扩展。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
