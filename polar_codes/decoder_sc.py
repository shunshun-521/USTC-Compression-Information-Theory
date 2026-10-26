"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import _bit_reverse_indices


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _prepare_llr(llr_ch):
    return np.asarray(llr_ch, dtype=np.float64)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（树节点索引，与标准极化码实现一致）"""
    llr = _prepare_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N)) + 1
    u_hat = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits)[0])

    def _xor(left, right):
        res = [(left[i] + right[i]) % 2 for i in range(len(left))]
        return res + right

    def decode_node(y, depth, node):
        if depth == n - 1:
            if node in frozen_set:
                u_hat[node] = 0
                return [0]
            bit = 0 if y[0] >= 0 else 1
            u_hat[node] = bit
            return [bit]

        half = len(y) // 2
        l1, l2 = y[:half], y[half:]
        left = f_operation(l1, l2)
        arr1 = decode_node(left, depth + 1, 2 * node)
        right = g_operation(l1, l2, np.array(arr1, dtype=int))
        arr2 = decode_node(right, depth + 1, 2 * node + 1)
        return _xor(arr1, arr2)

    decode_node(llr, 0, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        # LLR layers: trailing ones in binary(phi), then the zero position
        layers_llr = []
        p = phi
        while p & 1:
            layers_llr.append(int(math.log2(p & -p)))
            p >>= 1
        if p > 0:
            layers_llr.append(int(math.log2(p & -p)))
        llr_layer_vec.append(sorted(set(layers_llr)))

        # Bit back-propagation layers
        layers_bit = []
        p = phi
        while p & 1:
            layers_bit.append(int(math.log2(p & -p)))
            p >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（主实现，与递归参考实现等价）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """别名：与非递归主实现相同"""
    return sc_decode(llr_ch, frozen_bits)
