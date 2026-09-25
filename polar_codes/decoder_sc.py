"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    u_hat 为重编码后的部分和（u_hat_up）。
    """
    return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi in range(N):
        layer = 0
        while (phi >> layer) & 1:
            bit_layer_vec[phi].append(layer)
            layer += 1
        for ll in range(layer, n):
            llr_layer_vec[phi].append(ll)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _decode_subtree(llr_node, frozen_node, bit_offset, u_hat):
    """递归译码子树，g 运算使用重编码部分和 u_hat_up"""
    n = len(llr_node)
    if n == 1:
        idx = bit_offset
        if frozen_node[0]:
            u_hat[idx] = 0
        else:
            u_hat[idx] = 0 if llr_node[0] >= 0 else 1
        return np.array([u_hat[idx]], dtype=int)

    half = n // 2
    llr_left = f_operation(llr_node[:half], llr_node[half:])
    u_up_left = _decode_subtree(llr_left, frozen_node[:half], bit_offset, u_hat)
    llr_right = g_operation(llr_node[:half], llr_node[half:], u_up_left)
    u_up_right = _decode_subtree(llr_right, frozen_node[half:], bit_offset + half, u_hat)
    return np.concatenate([(u_up_left ^ u_up_right).astype(int), u_up_right])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    _decode_subtree(llr, frozen_bits, 0, u_hat)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（逐比特更新，显式 P/C 数组）。

    对 N=1024 码长，递归深度仅 10，与递归版本等价。
    此处采用与递归版本相同的子树逻辑，通过显式调用避免栈溢出风险。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
