"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_partial):
    """g 运算，u_partial 为部分和"""
    return (1 - 2 * u_partial) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def recurse(llr_in, offset):
        m = len(llr_in)
        if m == 1:
            idx = offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_in[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=int)

        half = m // 2
        l1, l2 = llr_in[:half], llr_in[half:]
        left_llr = f_operation(l1, l2)
        u_left = recurse(left_llr, offset)
        right_llr = g_operation(l1, l2, u_left)
        u_right = recurse(right_llr, offset + half)
        return np.concatenate([(u_left ^ u_right), u_right])

    recurse(llr, 0)
    return u_hat


def _b_check(layer, idx):
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, s):
    """更新部分和数组 s"""
    if _b_check(layer - 1, idx):
        s[layer, idx] = s[layer - 1, idx]
    else:
        half = 1 << (layer - 1)
        left_idx = idx
        right_idx = idx + half
        if s[layer - 1, left_idx] == -1:
            _s_updater(layer - 1, left_idx, s)
        if s[layer - 1, right_idx] == -1:
            _s_updater(layer - 1, right_idx, s)
        s[layer, idx] = s[layer - 1, left_idx] ^ s[layer - 1, right_idx]


def _compute_llr(layer, idx, llrs, s):
    """惰性计算 LLR"""
    if llrs[layer, idx] != -np.inf:
        return llrs[layer, idx]

    if _b_check(layer, idx) == 0:
        half = 1 << layer
        llrs[layer, idx] = f_operation(
            _compute_llr(layer + 1, idx, llrs, s),
            _compute_llr(layer + 1, idx + half, llrs, s),
        )
    else:
        half = 1 << layer
        left_idx = idx - half
        if layer > 0:
            _s_updater(layer, left_idx, s)
        llrs[layer, idx] = g_operation(
            _compute_llr(layer + 1, left_idx, llrs, s),
            _compute_llr(layer + 1, idx, llrs, s),
            s[layer, left_idx],
        )
    return llrs[layer, idx]


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容接口）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        pp = phi
        for layer in range(n):
            if (pp & 1) == 0:
                layers.append(layer)
            pp >>= 1
        llr_layer_vec.append(layers)
        if phi % 2 == 0:
            bit_layer_vec.append(list(range(n)))
        else:
            layers_b = []
            pp = phi
            for layer in range(n):
                if (pp & 1) == 1:
                    layers_b.append(layer)
                pp >>= 1
            bit_layer_vec.append(layers_b)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（惰性 LLR 计算，O(N log N)）。
    frozen_bits: True 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr_ch
    s = np.full((n + 1, N), -1, dtype=np.int8)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        if frozen_bits[phi]:
            s[0, phi] = 0
            llrs[0, phi] = np.inf
            u_hat[phi] = 0
        else:
            llrs[0, phi] = _compute_llr(0, phi, llrs, s)
            u_hat[phi] = 1 if llrs[0, phi] < 0 else 0
            s[0, phi] = u_hat[phi]

    return u_hat
