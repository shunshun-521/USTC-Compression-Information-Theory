"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from channel import prepare_channel_llr
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    log-domain f 运算（box-plus），支持向量化。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return _vec_logdomain_sum(La + Lb, 0.0) - _vec_logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    """标量 log(exp(x) + exp(y))"""
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _vec_logdomain_sum(x, y):
    """向量化 log(exp(x) + exp(y))"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    larger = np.maximum(x, y)
    smaller = np.minimum(x, y)
    return larger + np.log1p(np.exp(smaller - larger))


def _active_llr_level(i, n):
    """找 i 的二进制表示中第一个 1 的位置（从高位起）"""
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
    """找 i 的二进制表示中第一个 0 的位置（从高位起）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_reversed(i, n):
    """比特倒序"""
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，按比特索引递归）"""
    llr = prepare_channel_llr(llr_ch, len(llr_ch))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr
    decode_order = [_bit_reversed(i, n) for i in range(N)]

    def decode_step(k):
        if k >= N:
            return
        l = decode_order[k]
        _update_llrs(L, B, l, n, N)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)
        decode_step(k + 1)

    decode_step(0)
    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        br_phi = _bit_reversed(phi, n)
        llr_start = n - _active_llr_level(br_phi, n)
        llr_layer_vec.append(list(range(llr_start, n)))
        bit_start = n - _active_bit_level(br_phi, n) + 1
        bit_layer_vec.append(list(range(n, bit_start - 1, -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs(L, B, l, n, N):
    """更新 LLR 树（因子图消息传递）"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n, N):
    """比特回传"""
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 直接输入（与编码器比特倒序输出对应），
    按比特倒序索引顺序译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr = prepare_channel_llr(llr_ch, N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    decode_order = [_bit_reversed(i, n) for i in range(N)]

    for l in decode_order:
        _update_llrs(L, B, l, n, N)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        _update_bits(B, l, n, N)

    return B[:, n].astype(int)
