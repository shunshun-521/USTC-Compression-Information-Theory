"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

INF = -np.inf


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _b_check(layer, idx):
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, s):
    if _b_check(layer - 1, idx):
        s[layer, idx] = s[layer - 1, idx]
    else:
        if s[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, s)
        sibling = idx + (1 << (layer - 1))
        if s[layer - 1, sibling] == -1:
            _s_updater(layer - 1, sibling, s)
        s[layer, idx] = s[layer - 1, idx] ^ s[layer - 1, sibling]


def _li(layer, idx, llrs, s, n):
    if llrs[layer, idx] != INF:
        return llrs[layer, idx]

    if _b_check(layer, idx) == 0:
        llrs[layer, idx] = f_operation(
            _li(layer + 1, idx, llrs, s, n),
            _li(layer + 1, idx + (1 << layer), llrs, s, n),
        )
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), s)
        llrs[layer, idx] = g_operation(
            _li(layer + 1, idx - (1 << layer), llrs, s, n),
            _li(layer + 1, idx, llrs, s, n),
            s[layer, idx - (1 << layer)],
        )
    return llrs[layer, idx]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, depth, bit_offset):
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
        decode_node(llr_left, depth - 1, bit_offset)
        llr_right = g_operation(
            llr_node[:half], llr_node[half:], u_hat[bit_offset:bit_offset + half]
        )
        decode_node(llr_right, depth - 1, bit_offset + half)

    decode_node(llr, int(math.log2(N)), 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 使用）。"""
    n = int(np.log2(N))
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n)) for _ in range(N)]
    lambda_offset = list(range(n + 1))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（lazy LLR 计算）。"""
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=int)

    llrs = np.full((n + 1, N), INF, dtype=np.float64)
    llrs[n, :] = llr_ch.copy()
    s = np.full((n + 1, N), -1, dtype=int)
    u_hat = np.zeros(N, dtype=int)

    for idx in range(N):
        llr_val = _li(0, idx, llrs, s, n)
        if frozen_bits[idx]:
            u_hat[idx] = 0
            s[0, idx] = 0
        else:
            u_hat[idx] = 0 if llr_val >= 0 else 1
            s[0, idx] = u_hat[idx]

    return u_hat
