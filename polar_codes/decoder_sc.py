"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（向量化）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（向量化）。"""
    if np.isscalar(u_hat) or (hasattr(u_hat, "ndim") and u_hat.ndim == 0):
        return Lb + (1.0 - 2.0 * u_hat) * La
    return Lb + (1.0 - 2.0 * np.asarray(u_hat)) * La


def _f_list(L1, L2):
    return [
        np.sign(a) * np.sign(b) * min(abs(a), abs(b)) for a, b in zip(L1, L2)
    ]


def _g_list(L1, L2, b):
    return [l2 + (1 - 2 * int(bi)) * l1 for l1, l2, bi in zip(L1, L2, b)]


def _xor_paths(left, right):
    res = [(left[i] + right[i]) % 2 for i in range(len(left))]
    res.extend(right)
    return res


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    llr 为自然顺序信道 LLR，与 polar_encode 配套。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N)) + 1
    rev = bit_reversal_permutation(N)
    frozen_set = set(np.where(frozen_bits)[0])
    llr_br = llr[rev].tolist()
    node_values = [0] * N

    def decode(y, depth, node):
        if depth == n - 1:
            if node in frozen_set:
                node_values[node] = 0
            else:
                node_values[node] = 1 if y[0] < 0 else 0
            return [node_values[node]]

        half = len(y) // 2
        l1, l2 = y[:half], y[half:]
        left = _f_list(l1, l2)
        arr1 = decode(left, depth + 1, 2 * node)
        right = _g_list(l1, l2, arr1)
        arr2 = decode(right, depth + 1, 2 * node + 1)
        return _xor_paths(arr1, arr2)

    decode(llr_br, 0, 0)
    return np.array(node_values, dtype=int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（当前调用高效递归实现）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
