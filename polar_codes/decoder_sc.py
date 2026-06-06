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
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _is_g_node(layer, index):
    """判断因子图节点类型：0=f，1=g"""
    return (index // (1 << layer)) % 2


def _update_partial_sum(layer, index, s, n):
    """更新部分和树，供 g 节点使用"""
    if _is_g_node(layer - 1, index):
        s[layer, index] = s[layer - 1, index]
        return

    left = index
    right = index + (1 << (layer - 1))
    if s[layer - 1, left] == -1:
        _update_partial_sum(layer - 1, left, s, n)
    if s[layer - 1, right] == -1:
        _update_partial_sum(layer - 1, right, s, n)
    s[layer, index] = s[layer - 1, left] ^ s[layer - 1, right]


def _lazy_llr(layer, index, llrs, s, n):
    """惰性计算 LLR 树节点值"""
    if llrs[layer, index] != -np.inf:
        return llrs[layer, index]

    if _is_g_node(layer, index) == 0:
        llrs[layer, index] = f_operation(
            _lazy_llr(layer + 1, index, llrs, s, n),
            _lazy_llr(layer + 1, index + (1 << layer), llrs, s, n),
        )
    else:
        if layer > 0:
            _update_partial_sum(layer, index - (1 << layer), s, n)
        llrs[layer, index] = g_operation(
            _lazy_llr(layer + 1, index - (1 << layer), llrs, s, n),
            _lazy_llr(layer + 1, index, llrs, s, n),
            s[layer, index - (1 << layer)],
        )
    return llrs[layer, index]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
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
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（接口兼容）。
    """
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        tmp = phi
        layer = 0
        while layer < n:
            if tmp % 2 == 0:
                llr_layers.append(layer)
                tmp //= 2
                layer += 1
            else:
                break
        if not llr_layers or llr_layers[-1] != n - 1:
            llr_layers.append(n - 1)
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        tmp = phi
        layer = 0
        while layer < n:
            if tmp % 2 == 1:
                bit_layers.append(layer)
                tmp //= 2
                layer += 1
            else:
                break
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（惰性 LLR 更新实现）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    s = np.full((n + 1, N), -1, dtype=int)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
        llrs[n, :] = llr_ch

        if frozen_bits[phi]:
            u_hat[phi] = 0
            s[0, phi] = 0
            llrs[0, phi] = np.inf
        else:
            llr_val = _lazy_llr(0, phi, llrs, s, n)
            u_hat[phi] = 0 if llr_val >= 0 else 1
            s[0, phi] = u_hat[phi]

    return u_hat


def prepare_channel_llr(llr_ch):
    """
    将信道 LLR 重排为译码器所需顺序。
    编码端含比特倒序置换，因此接收 LLR 需做相同重排。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return llr_ch[br]


def sc_llr_to_bit(llr_val):
    """根据 LLR 判决比特"""
    return 0 if llr_val >= 0 else 1


def path_metric_update(pm, llr_val, u_bit):
    """路径度量更新：与 LLR 不一致时加 |LLR| 惩罚"""
    penalty = 0.0 if u_bit == sc_llr_to_bit(llr_val) else abs(llr_val)
    return pm + penalty
