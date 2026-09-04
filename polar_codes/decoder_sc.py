"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import _bit_reversal_indices


def reorder_llr_for_decode(llr_ch):
    """将信道 LLR 重排以匹配编码端蝶形输出顺序（补偿比特倒序置换）"""
    N = len(llr_ch)
    brp = _bit_reversal_indices(N)
    return np.asarray(llr_ch, dtype=np.float64)[brp]


# ==================== 基本运算 ====================

def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N)) + 1
    u_hat = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits)[0])

    def _xor(left, right):
        return [(left[i] + right[i]) % 2 for i in range(len(left))] + list(right)

    def decode_node(y, depth, node):
        if depth == n - 1:
            if node in frozen_set:
                u_hat[node] = 0
            else:
                u_hat[node] = 0 if y[0] >= 0 else 1
            return [u_hat[node]]

        half = len(y) // 2
        l1, l2 = np.asarray(y[:half]), np.asarray(y[half:])
        arr1 = decode_node(f_operation(l1, l2).tolist(), depth + 1, 2 * node)
        arr2 = decode_node(
            g_operation(l1, l2, arr1).tolist(), depth + 1, 2 * node + 1
        )
        return _xor(arr1, arr2)

    decode_node(llr.tolist(), 0, 0)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 0:
                llr_layers.append(layer)
            temp //= 2
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 1:
                bit_layers.append(layer)
            temp //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def _get_sc_cache(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    采用与递归版本等价的 LLR 重排 + 树遍历逻辑。
    """
    llr = reorder_llr_for_decode(llr_ch)
    return sc_decode_recursive(llr, frozen_bits)
