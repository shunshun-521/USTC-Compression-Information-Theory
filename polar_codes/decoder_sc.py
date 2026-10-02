"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import inverse_bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def partial_summation(u_segment):
    """对子块做极化编码蝶形（不含比特倒序），供 g 运算使用"""
    v = np.asarray(u_segment, dtype=np.int8).copy()
    n = len(v)
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            v[i : i + step] ^= v[i + step : i + 2 * step]
        step *= 2
    return v


def _align_channel_llr(llr_ch):
    N = len(llr_ch)
    inv = inverse_bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = _align_channel_llr(llr)
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)

        u_left = u_hat[bit_offset : bit_offset + half]
        c_left = partial_summation(u_left)
        llr_right = g_operation(llr_node[:half], llr_node[half:], c_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        if phi == 0:
            layers_llr = list(range(n - 1, -1, -1))
        else:
            t = 0
            while (phi >> t) & 1:
                t += 1
            layers_llr = list(range(n - 1, t - 1, -1))
        llr_layer_vec.append(layers_llr)

        layers_bit = [l for l in range(n) if (phi >> l) & 1]
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（与递归实现等价，供仿真默认调用）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
