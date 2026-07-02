"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（惰性 LLR 计算，高效实现）
"""
import numpy as np

from channel import prepare_channel_llr


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def _b_check(layer, index):
    return (index // (1 << layer)) % 2


def _s_updater(layer, index, bits):
    if _b_check(layer - 1, index):
        bits[layer, index] = bits[layer - 1, index]
    else:
        if bits[layer - 1, index] == -1:
            _s_updater(layer - 1, index, bits)
        partner = index + (1 << (layer - 1))
        if bits[layer - 1, partner] == -1:
            _s_updater(layer - 1, partner, bits)
        bits[layer, index] = bits[layer - 1, index] ^ bits[layer - 1, partner]


def _lazy_llr(layer, index, llrs, bits):
    if llrs[layer, index] != -np.inf:
        return llrs[layer, index]
    if _b_check(layer, index) == 0:
        llrs[layer, index] = f_operation(
            _lazy_llr(layer + 1, index, llrs, bits),
            _lazy_llr(layer + 1, index + (1 << layer), llrs, bits),
        )
    else:
        left = index - (1 << layer)
        if layer > 0:
            _s_updater(layer, left, bits)
        llrs[layer, index] = g_operation(
            _lazy_llr(layer + 1, left, llrs, bits),
            _lazy_llr(layer + 1, index, llrs, bits),
            bits[layer, left],
        )
    return llrs[layer, index]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = prepare_channel_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def rec(llr_node, frozen_node):
        if len(llr_node) == 1:
            if frozen_node[0]:
                return np.array([0])
            return np.array([0 if llr_node[0] >= 0 else 1])
        half = len(llr_node) // 2
        u_left = rec(f_operation(llr_node[:half], llr_node[half:]), frozen_node[:half])
        u_right = rec(g_operation(llr_node[:half], llr_node[half:], u_left), frozen_node[half:])
        return np.concatenate([u_left, u_right])

    return rec(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：惰性 LLR 实现不依赖预计算表"""
    n = int(np.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（惰性 LLR 计算）。
    llr_ch 经 prepare_channel_llr 做比特倒序后与编码器配套。
    """
    llr_ch = prepare_channel_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr_ch
    bits = np.full((n + 1, N), -1, dtype=np.int8)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        if frozen_bits[phi]:
            bits[0, phi] = 0
            u_hat[phi] = 0
        else:
            llrs[0, phi] = _lazy_llr(0, phi, llrs, bits)
            u_hat[phi] = 1 if llrs[0, phi] < 0 else 0
            bits[0, phi] = u_hat[phi]

    return u_hat
