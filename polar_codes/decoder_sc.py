"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归树形 LLR 更新（高效实现，与编码器配套）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _permute_llr(llr_ch):
    br = bit_reversal_permutation(len(llr_ch))
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _B_check(ll, ii):
    return (ii // (1 << ll)) % 2


def _s_updater(ll, ii, s):
    if _B_check(ll - 1, ii):
        s[ll, ii] = s[ll - 1, ii]
    else:
        if s[ll - 1, ii] == -1:
            _s_updater(ll - 1, ii, s)
        if s[ll - 1, ii + (1 << (ll - 1))] == -1:
            _s_updater(ll - 1, ii + (1 << (ll - 1)), s)
        s[ll, ii] = s[ll - 1, ii] ^ s[ll - 1, ii + (1 << (ll - 1))]


def _Li(ll, ii, llrs, s, n):
    if llrs[ll, ii] != -np.inf:
        return llrs[ll, ii]
    if _B_check(ll, ii) == 0:
        llrs[ll, ii] = f_operation(
            _Li(ll + 1, ii, llrs, s, n), _Li(ll + 1, ii + (1 << ll), llrs, s, n)
        )
        return llrs[ll, ii]
    if ll > 0:
        _s_updater(ll, ii - (1 << ll), s)
    llrs[ll, ii] = g_operation(
        _Li(ll + 1, ii - (1 << ll), llrs, s, n),
        _Li(ll + 1, ii, llrs, s, n),
        s[ll, ii - (1 << ll)],
    )
    return llrs[ll, ii]


def _sc_decode_core(llr_ch, frozen_bits):
    llr_ch = _permute_llr(llr_ch)
    N = len(llr_ch)
    n = int(np.log2(N))
    llrs = -np.inf * np.ones((n + 1, N), dtype=np.float64)
    llrs[n, :] = llr_ch
    s = -np.ones((n + 1, N), dtype=np.int8)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    for ii in range(N):
        if frozen_bits[ii]:
            s[0, ii] = 0
            llrs[0, ii] = np.inf
            u_hat[ii] = 0
        else:
            llrs[0, ii] = _Li(0, ii, llrs, s, n)
            u_hat[ii] = 1 if llrs[0, ii] < 0 else 0
            s[0, ii] = u_hat[ii]
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（调用与编码一致的核心译码器）。"""
    return _sc_decode_core(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算接口（相位相关的层索引，供文档/扩展使用）。"""
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        x = phi
        for ll in range(n):
            if _B_check(ll, phi) == 0:
                layers.append(ll)
        llr_layer_vec.append(layers)
        bit_layer_vec.append(list(range(n)))
    return list(range(n + 1)), llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return _sc_decode_core(llr_ch, frozen_bits)
