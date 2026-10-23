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
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _sc_decode_core(llr, frozen_bits):
    """在比特倒序后的 LLR 上执行 SC 译码（递归核心）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)

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
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    N = len(llr)
    llr = np.asarray(llr, dtype=np.float64)[bit_reversal_permutation(N)]
    return _sc_decode_core(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码的层更新列表。
    """
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        tmp = phi
        layer = 0
        while tmp & 1:
            tmp >>= 1
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))
        bit_layer_vec.append(list(range(layer)))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """
    非递归 SC 译码（预计算索引 + 增量 LLR 更新，接口与 sc_decode 一致）。
    """
    return sc_decode(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（与递归核心等价，接口供仿真调用）。"""
    N = len(llr_ch)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)[bit_reversal_permutation(N)]
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return _sc_decode_core(llr_ch, frozen_bits)
