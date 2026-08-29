"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation, polar_encode_core


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return bit_reversal_permutation(1 << n)[i]


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


def _permute_channel_llr(llr_ch, N):
  # 信道 LLR（对应 polar_encode 输出序）-> 因子图自然序
    br = bit_reversal_permutation(N)
    return llr_ch[br]


def _sc_lazy(llr_channel, frozen_bits):
    """
    Lazy SC 译码（自然序相位，与 polar_encode_core 配套）。
  llrs 布局: (n+1, N)，最后一行是信道 LLR。
    """
    llr_channel = np.asarray(llr_channel, dtype=np.float64)
    N = len(llr_channel)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=int)

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr_channel
    s = np.full((n + 1, N), -1, dtype=np.int8)

    def b_check(ll, ii):
        return (ii // (1 << ll)) % 2

    def s_updater(ll, ii):
        if b_check(ll - 1, ii):
            s[ll, ii] = s[ll - 1, ii]
        else:
            if s[ll - 1, ii] == -1:
                s_updater(ll - 1, ii)
            if s[ll - 1, ii + (1 << (ll - 1))] == -1:
                s_updater(ll - 1, ii + (1 << (ll - 1)))
            s[ll, ii] = s[ll - 1, ii] ^ s[ll - 1, ii + (1 << (ll - 1))]

    def li(ll, ii):
        if llrs[ll, ii] != -np.inf:
            return llrs[ll, ii]
        if b_check(ll, ii) == 0:
            llrs[ll, ii] = f_operation(li(ll + 1, ii), li(ll + 1, ii + (1 << ll)))
        else:
            if ll > 0:
                s_updater(ll, ii - (1 << ll))
            llrs[ll, ii] = g_operation(
                li(ll + 1, ii - (1 << ll)), li(ll + 1, ii), s[ll, ii - (1 << ll)]
            )
        return llrs[ll, ii]

    u_hat = np.zeros(N, dtype=int)
    for ii in range(N):
        cur = li(0, ii)
        if frozen_bits[ii]:
            u_hat[ii] = 0
            s[0, ii] = 0
        else:
            u_hat[ii] = 1 if cur < 0 else 0
            s[0, ii] = u_hat[ii]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    x_core = polar_encode_core(np.zeros(len(llr), dtype=int))  # noqa: unused - ensure import
    llr_nat = _permute_channel_llr(np.asarray(llr, dtype=np.float64), len(llr))
    return _sc_lazy(llr_nat, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    polar_encode（含比特倒序）时自动置换 LLR；内部使用 lazy SC。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    llr_nat = _permute_channel_llr(llr_ch, N)
    return _sc_lazy(llr_nat, frozen_bits)


def precompute_sc_indices(N):
    """SCL 兼容占位（lazy SC 不使用）。"""
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]
