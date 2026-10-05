"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Permuted SCD，高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（递归参考实现）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（递归参考实现）"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    return x + np.log1p(np.exp(y - x)) if x > y else y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _lower_llr(btm, top, b):
    if b == 0:
        if np.isinf(top) or np.isinf(btm):
            return np.inf
        return top + btm
    return top - btm


def _bit_reversed(x, n):
    y = 0
    for i in range(n):
        if x & (1 << i):
            y |= 1 << (n - 1 - i)
    return y


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _safe_bit(v):
    return 0 if np.isnan(v) else int(v)


def _align_llr_to_source(llr_ch):
    """极化编码含比特倒序时，将信道 LLR 对齐到源向量 u 的索引"""
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br]


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while p & 1:
            llr_layers.append(len(llr_layers))
            p >>= 1
        llr_layer_vec.append(llr_layers)
        start = len(llr_layers) if phi % 2 == 1 else 0
        bit_layer_vec.append(list(range(start, n)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = _align_llr_to_source(np.asarray(llr_ch, dtype=np.float64))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_left(llr_node, bit_offset, length):
        if length == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return
        half = length // 2
        llr_l = f_operation(llr_node[:half], llr_node[half:])
        decode_left(llr_l, bit_offset, half)
        u_l = u_hat[bit_offset : bit_offset + half]
        llr_r = g_operation(llr_node[:half], llr_node[half:], u_l)
        decode_left(llr_r, bit_offset + half, half)

    decode_left(llr, 0, N)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 Permuted SCD（对数域）"""
    from scd_core import scd

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    llr = _align_llr_to_source(llr_ch)
    frozen_set = set(np.where(frozen_bits)[0])
    return scd(llr, frozen_set, n).astype(np.int8)
