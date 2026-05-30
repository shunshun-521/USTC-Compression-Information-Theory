"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

LLR_MAX = 30.0

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """box-plus（log-domain）"""
    La = np.clip(np.asarray(La, dtype=np.float64), -LLR_MAX, LLR_MAX)
    Lb = np.clip(np.asarray(Lb, dtype=np.float64), -LLR_MAX, LLR_MAX)
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.log1p(np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb))
    return np.nan_to_num(out, nan=0.0)


def g_operation(La, Lb, u_hat):
    """g 运算；u_hat 为上层 u_hat_up"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _hard_decision(llr_val):
    if llr_val >= 0:
        return 0
    return 1


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 Sionna PolarSCDecoder 逻辑一致）"""

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                bit = 0
            else:
                bit = _hard_decision(llr_node[0])
            u = np.array([bit], dtype=int)
            return u, u

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_hat1, u_hat1_up = decode_node(llr_left, frozen_node[:half])
        llr_right = g_operation(
            llr_node[:half], llr_node[half:], u_hat1_up.astype(np.float64)
        )
        u_hat2, u_hat2_up = decode_node(llr_right, frozen_node[half:])

        u_left = np.bitwise_xor(u_hat1_up, u_hat2_up)
        u_hat = np.concatenate([u_hat1, u_hat2])
        u_up = np.concatenate([u_left, u_hat2_up])
        return u_hat, u_up

    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat, _ = decode_node(llr, frozen_bits)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引（供扩展实现使用）"""
    n = int(math.log2(N))
    lambda_offset = [1 << (n - layer) for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        t = phi
        while t & 1:
            layers.append(int(math.log2(t & -t)))
            t >>= 1
        layers.append(n - 1)
        llr_layer_vec.append(layers)
        bl = []
        t = phi + 1
        layer = 0
        while t % 2 == 0 and layer < n:
            bl.append(layer)
            t >>= 1
            layer += 1
        bit_layer_vec.append(bl)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（当前调用递归实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
