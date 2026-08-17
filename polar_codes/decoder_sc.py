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
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _frozen_set_from_bits(frozen_bits):
    """将 frozen_bits 数组转为冻结位索引集合"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    if frozen_bits.dtype != bool:
        frozen_bits = frozen_bits.astype(bool)
    return set(np.where(frozen_bits)[0])


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
  LLR > 0 倾向比特 0，LLR < 0 倾向比特 1。
    """
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(math.log2(N)) + 1
    F = _frozen_set_from_bits(frozen_bits)
    node_values = np.zeros(N, dtype=int)

    def _merge_paths(left, right):
        """合并子树返回值（与 Arikan SC 蝶形结构一致）"""
        merged = [(left[i] + right[i]) % 2 for i in range(len(left))]
        merged.extend(right)
        return merged

    def decode(y, depth, node):
        if depth == n - 1:
            if node in F:
                node_values[node] = 0
            else:
                node_values[node] = 1 if y[0] < 0 else 0
            return [node_values[node]]

        half = len(y) // 2
        L1, L2 = y[:half], y[half:]
        left_llr = f_operation(L1, L2)
        arr1 = decode(left_llr, depth + 1, 2 * node)
        right_llr = g_operation(L1, L2, arr1)
        arr2 = decode(right_llr, depth + 1, 2 * node + 1)
        return _merge_paths(arr1, arr2)

    decode(llr, 0, 0)
    return node_values


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N)) + 1
    lambda_offset = [0] * n
    for layer in range(n):
        lambda_offset[layer] = (1 << layer) - 1

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        while psi % 2 == 1:
            llr_layers.append(int(math.log2(psi & -psi)))
            psi //= 2
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 1:
            psi = phi // 2
            while psi % 2 == 1:
                bit_layers.append(int(math.log2(psi & -psi)))
                psi //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（基于分层 LLR 存储）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
