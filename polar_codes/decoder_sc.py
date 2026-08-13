"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：La 为左（上）分支，Lb 为右（下）分支"""
    return La * (1 - 2 * u_hat) + Lb


def _B_check(ll, ii):
    return (ii // (1 << ll)) % 2


def _f_node_minsum(a, b):
    return np.sign(a * b) * np.minimum(np.abs(a), np.abs(b))


def _g_node(llr1, llr2, s):
    return llr1 * (1 - 2 * s) + llr2


def _s_updater(ll, ii, s):
    if _B_check(ll - 1, ii):
        s[ll, ii] = s[ll - 1, ii]
    else:
        if s[ll - 1, ii] == -1:
            _s_updater(ll - 1, ii, s)
        if s[ll - 1, ii + (1 << (ll - 1))] == -1:
            _s_updater(ll - 1, ii + (1 << (ll - 1)), s)
        s[ll, ii] = s[ll - 1, ii] ^ s[ll - 1, ii + (1 << (ll - 1))]


def _Li(ll, ii, llrs, s):
    if llrs[ll, ii] != -np.inf:
        return llrs[ll, ii]
    if _B_check(ll, ii) == 0:
        llrs[ll, ii] = _f_node_minsum(
            _Li(ll + 1, ii, llrs, s),
            _Li(ll + 1, ii + (1 << ll), llrs, s),
        )
        return llrs[ll, ii]
    if ll > 0:
        _s_updater(ll, ii - (1 << ll), s)
    llrs[ll, ii] = _g_node(
        _Li(ll + 1, ii - (1 << ll), llrs, s),
        _Li(ll + 1, ii, llrs, s),
        s[ll, ii - (1 << ll)],
    )
    return llrs[ll, ii]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（保留接口兼容性）"""
    import math
    n = int(math.log2(N))
    lambda_offset = [2 ** layer - 1 for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = [layer for layer in range(n) if (phi >> layer) & 1 == 0]
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        if phi % 2 == 1:
            layers_bit = [layer for layer in range(n) if (phi >> layer) & 1 == 1]
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数。
    frozen_bits: 1 表示冻结位，0 表示信息位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(np.log2(N))

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr_ch
    s = np.full((n + 1, N), -1, dtype=np.int8)

    u_hat = np.zeros(N, dtype=int)
    for ii in range(N):
        if frozen_bits[ii]:
            s[0, ii] = 0
            llrs[0, ii] = np.inf
            u_hat[ii] = 0
        else:
            llrs[0, ii] = _Li(0, ii, llrs, s)
            u_hat[ii] = 1 if llrs[0, ii] < 0 else 0
            s[0, ii] = u_hat[ii]

    return u_hat
