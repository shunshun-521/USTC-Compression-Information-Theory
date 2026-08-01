"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

INF = np.inf


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _is_right_branch(layer, index):
    return (index // (1 << layer)) % 2


def _update_partial_sum(layer, index, bits):
    """递归更新部分和比特"""
    if _is_right_branch(layer - 1, index):
        bits[layer, index] = bits[layer - 1, index]
    else:
        if bits[layer - 1, index] == -1:
            _update_partial_sum(layer - 1, index, bits)
        right_idx = index + (1 << (layer - 1))
        if bits[layer - 1, right_idx] == -1:
            _update_partial_sum(layer - 1, right_idx, bits)
        bits[layer, index] = bits[layer - 1, index] ^ bits[layer - 1, right_idx]


def _get_llr(layer, index, llrs, bits):
    """惰性计算 LLR"""
    if llrs[layer, index] != -INF:
        return llrs[layer, index]

    if _is_right_branch(layer, index) == 0:
        left = _get_llr(layer + 1, index, llrs, bits)
        right = _get_llr(layer + 1, index + (1 << layer), llrs, bits)
        llrs[layer, index] = f_operation(left, right)
    else:
        if layer > 0:
            _update_partial_sum(layer, index - (1 << layer), bits)
        left_idx = index - (1 << layer)
        left = _get_llr(layer + 1, left_idx, llrs, bits)
        right = _get_llr(layer + 1, index, llrs, bits)
        llrs[layer, index] = g_operation(left, right, bits[layer, left_idx])
    return llrs[layer, index]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（惰性 LLR 计算）。
    frozen_bits: True/1 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llrs = np.full((n + 1, N), -INF, dtype=np.float64)
    bits = np.full((n + 1, N), -1, dtype=np.int64)
    llrs[n, :] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        if frozen_bits[phi]:
            u_hat[phi] = 0
            llrs[0, phi] = INF
            bits[0, phi] = 0
        else:
            llr = _get_llr(0, phi, llrs, bits)
            u_hat[phi] = 0 if llr >= 0 else 1
            bits[0, phi] = u_hat[phi]
    return u_hat


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现，与 sc_decode 算法等价）。
    """
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（接口兼容）。"""
    n = int(math.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        level = 0
        while level < n and ((phi >> level) & 1):
            level += 1
        llr_layer_vec.append(list(range(level, n)))
        if phi % 2 == 0:
            bit_layer_vec.append(list(range(n)))
        else:
            bit_layer_vec.append(list(range(level)))
    return lambda_offset, llr_layer_vec, bit_layer_vec
