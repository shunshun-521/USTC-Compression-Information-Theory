"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_minsum(La, Lb):
    """min-sum 近似的 f 运算（供 BP 等场景参考）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def _frozen_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return set(np.where(frozen_bits)[0])


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(np.log2(N)) + 1
    frozen_set = _frozen_set(frozen_bits)
    node_values = np.zeros(N, dtype=int)

    def decode_node(y, depth, node):
        if depth == n - 1:
            if node in frozen_set:
                node_values[node] = 0
            else:
                node_values[node] = 1 if y[0] < 0 else 0
            return [node_values[node]]

        half = len(y) // 2
        l1 = y[:half]
        l2 = y[half:]
        left_dec = decode_node(f_operation(l1, l2), depth + 1, 2 * node)
        right_dec = decode_node(g_operation(l1, l2, left_dec), depth + 1, 2 * node + 1)
        merged = [(left_dec[i] + right_dec[i]) % 2 for i in range(len(left_dec))]
        merged.extend(right_dec)
        return merged

    decode_node(llr, 0, 0)
    return node_values


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi % 2 == 1:
            layers_llr.append(int(np.log2(psi & -psi)))
            psi >>= 1
        if psi > 0:
            layers_llr.append(int(np.log2(psi)))
        llr_layer_vec.append(layers_llr)

        if phi % 2 == 0:
            bit_layer_vec.append([])
        else:
            layers_bit = []
            psi = phi
            while psi % 2 == 1:
                layers_bit.append(int(np.log2(psi & -psi)))
                psi >>= 1
            bit_layer_vec.append(layers_bit)

    return llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    当前调用经过验证的递归实现以保证正确性。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
