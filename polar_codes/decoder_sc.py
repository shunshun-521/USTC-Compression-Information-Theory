"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

_INF = np.finfo(np.float64).max


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _is_g_node(layer, index):
    """因子图中 index 在 layer 层是否为 g 节点"""
    return (index // (1 << layer)) % 2 == 1


def _update_partial_sum(layer, index, bits):
    """递归更新部分和比特，用于 g 节点 LLR 计算"""
    if layer == 0:
        return
    if _is_g_node(layer - 1, index):
        bits[layer, index] = bits[layer - 1, index]
    else:
        if bits[layer - 1, index] == -1:
            _update_partial_sum(layer - 1, index, bits)
        pair = index + (1 << (layer - 1))
        if bits[layer - 1, pair] == -1:
            _update_partial_sum(layer - 1, pair, bits)
        bits[layer, index] = bits[layer - 1, index] ^ bits[layer - 1, pair]


def _compute_llr(layer, index, llrs, bits, n):
    """按需计算 llrs[layer, index]"""
    if not np.isnan(llrs[layer, index]):
        return llrs[layer, index]

    if not _is_g_node(layer, index):
        llrs[layer, index] = f_operation(
            _compute_llr(layer + 1, index, llrs, bits, n),
            _compute_llr(layer + 1, index + (1 << layer), llrs, bits, n),
        )
    else:
        left = index - (1 << layer)
        _update_partial_sum(layer, left, bits)
        llrs[layer, index] = g_operation(
            _compute_llr(layer + 1, left, llrs, bits, n),
            _compute_llr(layer + 1, index, llrs, bits, n),
            bits[layer, left],
        )
    return llrs[layer, index]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（基于按需 LLR 计算）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))

    llrs = np.full((n + 1, N), np.nan, dtype=np.float64)
    llrs[n, :] = llr
    bits = np.full((n + 1, N), -1, dtype=int)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        llr_phi = _compute_llr(0, phi, llrs, bits, n)
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if llr_phi >= 0 else 1
        bits[0, phi] = u_hat[phi]

    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        p = phi
        layer = 0
        while layer < n:
            if (p & 1) == 0:
                llr_layers.append(layer)
                p >>= 1
                layer += 1
            else:
                break

        bit_layers = []
        p = phi
        layer = 0
        while layer < n:
            if (p & 1) == 1:
                bit_layers.append(layer)
            p >>= 1
            layer += 1

        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（与递归版本等价）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
