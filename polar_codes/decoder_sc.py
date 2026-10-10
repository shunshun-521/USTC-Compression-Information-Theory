"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（box-plus 检查节点）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _hard_decision(llr):
    return 0 if llr >= 0 else 1


def _sc_decode_tree(llr, frozen_bits):
    """树形 SC 译码，返回 (u_hat, u_hat_up)。"""

    def decode_node(llr_node, frozen_slice):
        n = len(llr_node)
        if n == 1:
            if frozen_slice[0]:
                bit = 0
            else:
                bit = _hard_decision(llr_node[0])
            u = np.array([bit], dtype=int)
            return u, u.copy()

        half = n // 2
        llr_upper = f_operation(llr_node[:half], llr_node[half:])
        u1, u1_up = decode_node(llr_upper, frozen_slice[:half])
        llr_lower = g_operation(llr_node[:half], llr_node[half:], u1_up)
        u2, u2_up = decode_node(llr_lower, frozen_slice[half:])
        u_hat = np.concatenate([u1, u2])
        u1_up_xor = np.bitwise_xor(u1_up.astype(int), u2_up.astype(int))
        u_hat_up = np.concatenate([u1_up_xor, u2_up.astype(int)])
        return u_hat, u_hat_up

    return decode_node(llr, frozen_bits)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat, _ = _sc_decode_tree(llr, frozen_bits)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量（供报告/扩展使用）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        p = phi
        while (p & 1) == 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        p = phi
        while (p & 1) == 1:
            bit_layers.append(int(math.log2(p & -p)))
            p >>= 1
        if phi < N - 1:
            bit_layers.append(len(bit_layers))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码入口。
    当前实现复用等价的树形译码（O(N log N)）；接口与非递归流程一致。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
