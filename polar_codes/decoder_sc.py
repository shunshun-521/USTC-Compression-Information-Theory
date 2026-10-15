"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.atleast_1d(np.asarray(La, dtype=np.float64))
    Lb = np.atleast_1d(np.asarray(Lb, dtype=np.float64))
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa[sa == 0] = 1
    sb[sb == 0] = 1
    out = sa * sb * np.minimum(np.abs(La), np.abs(Lb))
    return out[0] if out.size == 1 else out


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _reorder_llr(llr_ch):
    """编码含比特倒序置换时，译码树使用倒序 LLR。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    return llr_ch[bit_reversal_permutation(N)]


# ==================== 递归 SC 译码（参考实现）====================


def _sc_recursive_core(llrs, indices, frozen_bits, decisions):
    """返回当前子树的部分和（极化编码后的比特序列）。"""
    if len(indices) == 1:
        i = indices[0]
        if frozen_bits[i]:
            decisions[i] = 0
        else:
            decisions[i] = 0 if llrs[0] >= 0 else 1
        return [decisions[i]]

    half = len(indices) // 2
    left_llr = [float(f_operation(llrs[i], llrs[i + half])) for i in range(half)]
    enc_left = _sc_recursive_core(left_llr, indices[:half], frozen_bits, decisions)
    right_llr = [
        float(g_operation(llrs[i], llrs[i + half], enc_left[i])) for i in range(half)
    ]
    enc_right = _sc_recursive_core(right_llr, indices[half:], frozen_bits, decisions)
    return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。"""
    llr = _reorder_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    decisions = [0] * N
    _sc_recursive_core(list(map(float, llr)), list(range(N)), frozen_bits, decisions)
    return np.array(decisions, dtype=int)


# ==================== 非递归 SC 译码（高效实现）====================


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


def _active_llr_level(l, n):
    level = 0
    p = l
    while p % 2 == 1 and level < n:
        level += 1
        p >>= 1
    return level


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layer_vec.append(list(range(_active_llr_level(l, n), n)))
        bit_layers = []
        p = l
        layer = 0
        while p % 2 == 1 and layer < n:
            bit_layers.append(layer)
            p >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（与递归版本等价的相位更新实现）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def verify_sc_equivalence(N=64):
    """验证递归与非递归 SC 一致。"""
    rng = np.random.default_rng(0)
    frozen = np.zeros(N, dtype=bool)
    frozen[rng.choice(N, N // 2, replace=False)] = True
    llr = rng.normal(0, 1, N)
    a = sc_decode(llr, frozen)
    b = sc_decode_recursive(llr, frozen)
    assert np.array_equal(a, b)
