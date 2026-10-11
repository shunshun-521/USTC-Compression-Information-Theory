"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _hetsn_depth(N):
    return int(math.log2(N)) + 1


def _sc_decode_hetsn(channel_vals, frozen_bits):
    """
    与 encoder.polar_encode 配套的 SC 译码（HETSN 递归结构）。
    channel_vals: 与 BPSK 解调后符号成比例的实数（可用 LLR）。
    """
    y = np.asarray(channel_vals, dtype=np.float64)
    N = len(y)
    n = _hetsn_depth(N)
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
    node_values = [0] * N

    def decode(cur_y, d, node):
        if d == n - 1:
            if node in frozen_set:
                node_values[node] = 0
            else:
                node_values[node] = 1 if cur_y[0] < 0 else 0
            return [node_values[node]]

        half = len(cur_y) // 2
        L1 = cur_y[:half]
        L2 = cur_y[half:]
        left = f_operation(L1, L2)
        arr1 = decode(left, d + 1, 2 * node)
        right = g_operation(L1, L2, np.array(arr1))
        arr2 = decode(right, d + 1, 2 * node + 1)
        merged = [(arr1[i] + arr2[i]) % 2 for i in range(len(arr1))] + list(arr2)
        return merged

    decode(y, 0, 0)
    return np.array(node_values, dtype=int)


def _hetsn_llr_at_bit(channel_vals, frozen_bits, u_prefix, phi):
    """已知前缀比特时，计算比特 phi 的 LLR（叶节点度量）"""
    y = np.asarray(channel_vals, dtype=np.float64)
    N = len(y)
    n = _hetsn_depth(N)
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
    u_prefix = np.asarray(u_prefix, dtype=int)

    def bit_range(d, node):
        size = 2 ** (n - 1 - d)
        start = node * size
        return start, start + size

    def walk(cur_y, d, node):
        start, end = bit_range(d, node)
        if d == n - 1:
            return float(cur_y[0])
        half = len(cur_y) // 2
        L1 = cur_y[:half]
        L2 = cur_y[half:]
        left_node = 2 * node
        right_node = 2 * node + 1
        _, left_end = bit_range(d + 1, left_node)
        if phi < left_end:
            return walk(f_operation(L1, L2), d + 1, left_node)
        l_start, l_end = bit_range(d + 1, left_node)
        arr1 = []
        for b in range(l_start, l_end):
            if b in frozen_set:
                arr1.append(0)
            else:
                arr1.append(int(u_prefix[b]))
        right = g_operation(L1, L2, np.array(arr1))
        return walk(right, d + 1, right_node)

    return walk(y, 0, 0)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    return _sc_decode_hetsn(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口"""
    return _sc_decode_hetsn(llr_ch, frozen_bits)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归接口（当前与递归实现相同）"""
    return _sc_decode_hetsn(llr_ch, frozen_bits)


def sc_llr_at_bit(llr, frozen_bits, u_prefix, phi):
    """SCL 使用的逐比特 LLR"""
    return _hetsn_llr_at_bit(llr, frozen_bits, u_prefix, phi)


def precompute_sc_indices(N):
    """保留接口"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]
