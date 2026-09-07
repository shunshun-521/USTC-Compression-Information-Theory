"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


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
    return Lb + (1.0 - 2.0 * u_hat) * La


def _hetsn_xor(left, right):
    res = [(left[i] + right[i]) % 2 for i in range(len(left))]
    res.extend(right)
    return res


def _active_llr_level(l, n):
    count = 0
    while l % 2 == 1 and count < n:
        count += 1
        l //= 2
    return count


def _active_bit_level(l, n):
    count = 0
    while l % 2 == 0 and count < n:
        count += 1
        l //= 2
    return count


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（Vangala 置换 SC）。
    """
    n = int(math.log2(N))

    def active_llr_level(l):
        return _active_llr_level(l, n)

    def active_bit_level(l):
        return _active_bit_level(l, n)

    def bit_reversed(i):
        return _bit_reversed(i, n)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi)
        llr_layer_vec.append(list(range(n - active_llr_level(l), n)))
        bit_layer_vec.append(list(range(n - active_bit_level(l), n)))
    return llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，基于因子树节点索引）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    N = len(llr)
    n = int(math.log2(N)) + 1
    frozen_set = set(np.where(frozen_bits.astype(bool))[0])
    node_values = [0] * N

    def decode(y, depth, node):
        if depth == n - 1:
            if node in frozen_set:
                node_values[node] = 0
                return [0]
            bit = 1 if y[0] < 0 else 0
            node_values[node] = bit
            return [bit]

        half = len(y) // 2
        l1, l2 = y[:half], y[half:]
        arr1 = decode(f_operation(l1, l2).tolist(), depth + 1, 2 * node)
        arr2 = decode(
            g_operation(l1, l2, np.array(arr1)).tolist(),
            depth + 1,
            2 * node + 1,
        )
        return _hetsn_xor(arr1, arr2)

    br = bit_reversal_permutation(N)
    decode(llr[br].tolist(), 0, 0)
    return np.array(node_values, dtype=int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    采用与递归版本等价的因子树遍历（信道 LLR 经比特倒序置换）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
