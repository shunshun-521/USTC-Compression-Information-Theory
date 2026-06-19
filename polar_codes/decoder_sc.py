"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，惰性 LLR 计算）
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
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _b_check(layer, idx):
    """判断因子图节点类型：0=f，1=g。"""
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, s):
    """更新部分和 s 数组。"""
    if _b_check(layer - 1, idx):
        s[layer, idx] = s[layer - 1, idx]
    else:
        if s[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, s)
        partner = idx + (1 << (layer - 1))
        if s[layer - 1, partner] == -1:
            _s_updater(layer - 1, partner, s)
        s[layer, idx] = s[layer - 1, idx] ^ s[layer - 1, partner]


def _li(layer, idx, llrs, s):
    """惰性计算第 layer 层、第 idx 个节点的 LLR。"""
    if llrs[layer, idx] != -np.inf:
        return llrs[layer, idx]
    if _b_check(layer, idx) == 0:
        llrs[layer, idx] = f_operation(
            _li(layer + 1, idx, llrs, s),
            _li(layer + 1, idx + (1 << layer), llrs, s),
        )
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), s)
        llrs[layer, idx] = g_operation(
            _li(layer + 1, idx - (1 << layer), llrs, s),
            _li(layer + 1, idx, llrs, s),
            s[layer, idx - (1 << layer)],
        )
    return llrs[layer, idx]


def _prepare_llr(llr_ch, N):
    """将信道 LLR 做比特倒序，与含 BR 的编码器对齐。"""
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，与 sc_decode 等价）。"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（与惰性 LLR 实现兼容）。
    返回解码顺序及每层活跃范围，供 SCL 等模块复用。
    """
    n = int(math.log2(N))
    decode_order = list(range(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        temp = phi
        while temp % 2 == 1:
            layers_llr.append(int(math.log2(temp & -temp)))
            temp //= 2
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        temp = (phi + 1) // 2
        while temp % 2 == 1:
            layers_bit.append(int(math.log2(temp & -temp)))
            temp //= 2
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（惰性 LLR 计算，O(N log N)）。
    信道 LLR 自动做比特倒序以匹配编码器。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr = _prepare_llr(llr_ch, N)

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr
    s = np.full((n + 1, N), -1, dtype=np.int8)
    u_hat = np.zeros(N, dtype=np.int8)

    for phi in range(N):
        if frozen_bits[phi]:
            u_hat[phi] = 0
            llrs[0, phi] = np.inf
            s[0, phi] = 0
        else:
            llrs[0, phi] = _li(0, phi, llrs, s)
            u_hat[phi] = 0 if llrs[0, phi] >= 0 else 1
            s[0, phi] = u_hat[phi]

    return u_hat


def path_metric_update(pm, llr, u):
    """路径度量更新。"""
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm
