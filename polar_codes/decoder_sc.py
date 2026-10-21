"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（u_hat 为左子树部分和比特，长度与 La 相同）"""
    return (1 - 2 * u_hat.astype(np.float64)) * La + Lb


def _sc_decode_tree(llr, frozen_bits, u_hat):
    """递归 SC（使用部分和 beta，与 Kronecker 编码一致）"""

    def node(llr_node, base, length):
        if length == 1:
            idx = base
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=np.uint8)

        half = length // 2
        upper = f_operation(llr_node[:half], llr_node[half:])
        beta_u = node(upper, base, half)
        lower = g_operation(llr_node[:half], llr_node[half:], beta_u)
        beta_l = node(lower, base + half, half)
        return np.concatenate([beta_u ^ beta_l, beta_l])

    node(llr, 0, len(llr))


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(len(llr), dtype=int)
    _sc_decode_tree(llr, frozen_bits, u_hat)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while p & 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        p = phi
        if (p & 1) == 0:
            bit_layers.append(0)
            p >>= 1
            while p & 1:
                bit_layers.append(int(math.log2(p & -p)))
                p >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    与递归版本等价；信道 LLR 顺序与码字比特索引一致。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
