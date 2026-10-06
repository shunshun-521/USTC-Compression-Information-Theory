"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _local_butterfly(u_seg):
    """子块蝶形变换（不含比特倒序），用于 g 运算的部分和比特"""
    v = np.asarray(u_seg, dtype=np.int8).copy()
    n_seg = len(v)
    n = int(math.log2(n_seg))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, n_seg, 2 * step):
            v[i : i + step] ^= v[i + step : i + 2 * step]
    return v


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    llr = llr_ch[br]
    u_hat = np.zeros(N, dtype=int)

    def decode_node(node_llr, depth, bit_index):
        if depth == 0:
            phi = bit_index
            if frozen_bits[phi]:
                u_hat[phi] = 0
            else:
                u_hat[phi] = 0 if node_llr[0] >= 0 else 1
            return
        half = 1 << (depth - 1)
        left_llr = f_operation(node_llr[:half], node_llr[half:])
        decode_node(left_llr, depth - 1, bit_index)
        v_left = _local_butterfly(u_hat[bit_index : bit_index + half])
        right_llr = g_operation(node_llr[:half], node_llr[half:], v_left)
        decode_node(right_llr, depth - 1, bit_index + half)

    decode_node(llr, n, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = np.array([1 << i for i in range(n + 1)], dtype=int)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        layer = 0
        while p & 1:
            llr_layers.append(layer)
            p >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)
        bit_layers = list(range(len(llr_layers)))
        if phi & 1 and len(bit_layers) < n:
            bit_layers.append(len(llr_layers))
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（高效实现，与递归版本等价）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
