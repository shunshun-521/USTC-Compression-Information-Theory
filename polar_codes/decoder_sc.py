"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def _sign_pm(x):
    """符号函数：0 映射为 +1，避免 min-sum f 在 0 处退化"""
    s = np.sign(x)
    return np.where(s == 0, 1.0, s)


def f_operation(La, Lb):
    """min-sum 近似 f 运算（含零 LLR 退化处理）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    eps = 1e-12
    core = _sign_pm(La) * _sign_pm(Lb) * np.minimum(np.abs(La), np.abs(Lb))
    return np.where(np.abs(La) < eps, Lb, np.where(np.abs(Lb) < eps, La, core))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_llr(llr_seg, u_hat, phi, bit_start):
    """
    计算位置 phi 的 LLR（已知 u_hat[0:phi]）。
    llr_seg 为当前子向量信道 LLR，bit_start 为子向量在全局 u 中的起始索引。
    """
    n = len(llr_seg)
    if n == 1:
        return float(llr_seg[0])

    half = n // 2
    mid = bit_start + half

    if phi < mid:
        llr_left = f_operation(llr_seg[:half], llr_seg[half:])
        return _bit_llr(llr_left, u_hat, phi, bit_start)
    if phi > mid:
        u_left = u_hat[bit_start:mid]
        llr_right = g_operation(llr_seg[:half], llr_seg[half:], u_left)
        return _bit_llr(llr_right, u_hat, phi, mid)

    # phi == mid：右子树第一个比特
    u_left = u_hat[bit_start:mid]
    llr_right = g_operation(llr_seg[:half], llr_seg[half:], u_left)
    return _bit_llr(llr_right, u_hat, phi, mid)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                return np.array([0], dtype=np.int8)
            return np.array([0 if llr_node[0] >= 0 else 1], dtype=np.int8)

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left = decode_node(llr_left, frozen_node[:half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        u_right = decode_node(llr_right, frozen_node[half:])
        return np.concatenate([u_left, u_right])

    return decode_node(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（供报告/扩展使用）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    for phi in range(N):
        layers = [layer for layer in range(n) if (phi >> layer) & 1]
        llr_layer_vec.append(layers)

    bit_layer_vec = []
    for phi in range(N):
        layers = []
        if phi % 2 == 0:
            layer = 0
            p = phi
            while (p % 2 == 0) and layer < n:
                layers.append(layer)
                p >>= 1
                layer += 1
        bit_layer_vec.append(layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC（SCAN：单次 BP 迭代的因子图调度，与本项目编码器/信道约定一致）。
    """
    from decoder_bp import BPDecoder

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    u_hat, _ = BPDecoder(N, frozen_bits, max_iter=1).decode(llr_ch)
    return u_hat
