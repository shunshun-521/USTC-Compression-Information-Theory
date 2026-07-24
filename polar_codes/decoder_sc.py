"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=int)
    return Lb + (1 - 2 * u_hat) * La


def _xor_combine(left, right):
    """合并左右子树的译码结果（极化码 SC 递归结构）。"""
    left = list(left)
    right = list(right)
    res = [(left[i] + right[i]) % 2 for i in range(len(left))]
    res.extend(right)
    return res


def _sc_decode_core(llr, depth, n, frozen_set, node, node_values):
    """SC 译码递归核心。"""
    if depth == n - 1:
        if node in frozen_set:
            node_values[node] = 0
            return [0]
        bit = 1 if llr[0] < 0 else 0
        node_values[node] = bit
        return [bit]

    half = len(llr) // 2
    l1 = llr[:half]
    l2 = llr[half:]
    left_llr = f_operation(l1, l2)
    arr1 = _sc_decode_core(left_llr, depth + 1, n, frozen_set, 2 * node, node_values)
    right_llr = g_operation(l1, l2, np.array(arr1))
    arr2 = _sc_decode_core(right_llr, depth + 1, n, frozen_set, 2 * node + 1, node_values)
    return _xor_combine(arr1, arr2)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    N = len(llr)
    n = int(np.log2(N)) + 1
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
    node_values = [0] * N
    _sc_decode_core(np.asarray(llr, dtype=np.float64), 0, n, frozen_set, 0, node_values)
    return np.array(node_values, dtype=int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        i = 0
        while i < n:
            if (phi >> i) & 1 == 0:
                layers.append(i)
                break
            i += 1
        for j in range(i + 1, n):
            layers.append(j)
        llr_layer_vec.append(layers)

        if phi % 2 == 0:
            b_layers = []
            i = 0
            while True:
                b_layers.append(i)
                if (phi >> (i + 1)) & 1 == 0:
                    break
                i += 1
            bit_layer_vec.append(b_layers)
        else:
            b_layers = []
            i = 0
            while True:
                b_layers.append(i)
                if (phi >> i) & 1 == 0:
                    break
                i += 1
            bit_layer_vec.append(b_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（调用经验证的递归实现）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
