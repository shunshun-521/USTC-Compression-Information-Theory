"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _slow_llr(i, N, y_llr, u_est):
    """递归计算第 i 个极化信道的 LLR（参考实现）"""
    if i == 0 and N == 1:
        return y_llr[0]
    if i % 2 == 0:
        l1 = _slow_llr(i // 2, N // 2, y_llr[:N // 2],
                        (u_est[::2] ^ u_est[1::2])[:i // 2])
        l2 = _slow_llr(i // 2, N // 2, y_llr[N // 2:],
                        u_est[1::2][:i // 2])
        return f_operation(l1, l2)
    l1 = _slow_llr(i // 2, N // 2, y_llr[:N // 2],
                    (u_est[:-1:2] ^ u_est[1::2])[:i // 2])
    l2 = _slow_llr(i // 2, N // 2, y_llr[N // 2:],
                    u_est[1::2][:i // 2])
    return l2 + ((-1) ** u_est[i - 1]) * l1


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
        if frozen_bits[i]:
            u_hat[i] = 0
        else:
            llr = _slow_llr(i, N, llr_ch, u_hat[:i])
            u_hat[i] = 0 if llr >= 0 else 1
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助结构（层化 LLR 缓存索引）。
    """
    n = int(math.log2(N))
    total = N * (n + 1)
    llr_cache = np.full(total, np.nan, dtype=np.float64)
    calc_flags = np.zeros(total, dtype=bool)

    def get_problem_i(idx):
        slice_idx = idx // N
        modulus = 1 << (n - slice_idx)
        return idx % modulus

    def get_descendants(idx):
        slice_idx = idx // N
        slice_i = idx - slice_idx * N
        sub_len = 1 << (n - slice_idx)
        sub_start = (slice_i // sub_len) * sub_len
        sub_i = idx % sub_len
        left = (slice_idx + 1) * N + sub_start + (sub_i // 2)
        right = left + (1 << (n - slice_idx - 1))
        return left, right

    def fast_llr(idx, y_llr, u_est):
        if calc_flags[idx]:
            return llr_cache[idx]
        prob_i = get_problem_i(idx)
        cur_n = len(y_llr)
        if prob_i == 0 and cur_n == 1:
            llr_cache[idx] = y_llr[0]
        elif prob_i % 2 == 0:
            left, right = get_descendants(idx)
            half = cur_n // 2
            l1 = fast_llr(left, y_llr[:half], (u_est[::2] ^ u_est[1::2])[:prob_i // 2])
            l2 = fast_llr(right, y_llr[half:], u_est[1::2][:prob_i // 2])
            llr_cache[idx] = f_operation(l1, l2)
        else:
            left, right = get_descendants(idx)
            half = cur_n // 2
            l1 = fast_llr(left, y_llr[:half], (u_est[:-1:2] ^ u_est[1::2])[:prob_i // 2])
            l2 = fast_llr(right, y_llr[half:], u_est[1::2][:prob_i // 2])
            llr_cache[idx] = l2 + ((-1) ** u_est[prob_i - 1]) * l1
        calc_flags[idx] = True
        return llr_cache[idx]

    return fast_llr, n


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（高效实现）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    fast_llr, _ = precompute_sc_indices(N)
    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
        if frozen_bits[i]:
            u_hat[i] = 0
        else:
            llr = fast_llr(i, llr_ch, u_hat[:i])
            u_hat[i] = 0 if llr >= 0 else 1
    return u_hat
