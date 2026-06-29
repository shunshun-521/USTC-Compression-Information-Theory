"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def _logdomain_sum(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diff = y - x
    diff = np.clip(diff, -50.0, 50.0)
    rev_diff = np.clip(x - y, -50.0, 50.0)
    return np.where(x > y, x + np.log1p(np.exp(diff)), y + np.log1p(np.exp(rev_diff)))


def f_boxplus(La, Lb):
    """精确 box-plus f 运算（SC 译码内部使用）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    scalar = La.ndim == 0 and Lb.ndim == 0
    La = np.atleast_1d(La)
    Lb = np.atleast_1d(Lb)
    out = _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)
    return out.item() if scalar else out


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def _update_llrs(L, B, l, n, upper=f_boxplus):
    for s in range(n - _active_llr_level(l, n), n):
        block = 2 ** (s + 1)
        branch = block // 2
        for j in range(l, L.shape[0], block):
            if j % block < branch:
                L[j, s + 1] = float(upper(L[j, s], L[j + branch, s]))
            else:
                L[j, s + 1] = float(
                    g_operation(
                        L[j - branch, s], L[j, s], B[j - branch, s + 1]
                    )
                )


def _update_bits(B, l, n):
    if l < B.shape[0] / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block = 2**s
        branch = block // 2
        for j in range(l, -1, -block):
            if j % block >= branch:
                B[j - branch, s - 1] = int(B[j, s]) ^ int(B[j - branch, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits, use_minsum=False):
    """
    递归 SC 译码。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    upper = f_operation if use_minsum else f_boxplus
    N = len(llr)
    if N == 1:
        if frozen_bits[0]:
            return np.array([0], dtype=int)
        return np.array([0 if llr[0] >= 0 else 1], dtype=int)

    half = N // 2
    llr_left = upper(llr[0::2], llr[1::2])
    u_left = sc_decode_recursive(llr_left, frozen_bits[:half], use_minsum)
    llr_right = g_operation(llr[0::2], llr[1::2], u_left)
    u_right = sc_decode_recursive(llr_right, frozen_bits[half:], use_minsum)
    return np.concatenate([u_left, u_right])


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        tmp = phi
        for layer in range(n):
            if tmp % 2 == 0:
                llr_layers.append(layer)
            tmp >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        tmp = phi + 1
        for layer in range(n):
            if tmp % 2 == 1:
                bit_layers.append(layer)
            tmp >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits, use_minsum=False):
    """
    非递归 SC 译码主函数。
    """
    llr_ch = prepare_channel_llr(llr_ch, len(llr_ch))
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    N = len(llr_ch)
    n = int(math.log2(N))
    upper = f_operation if use_minsum else f_boxplus

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = bit_reversed_index(i, n)
        _update_llrs(L, B, l, n, upper=upper)
        if l in frozen_set:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]
        _update_bits(B, l, n)

    return u_hat


def prepare_channel_llr(llr_ch, N):
    """
    将信道 LLR 调整为与编码端比特倒序一致的译码器输入顺序。
    """
    n = int(math.log2(N))
    br = np.array([bit_reversed_index(i, n) for i in range(N)], dtype=int)
    inv_perm = np.argsort(br)
    return np.asarray(llr_ch, dtype=np.float64)[inv_perm]
