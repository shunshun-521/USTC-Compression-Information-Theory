"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（check node / upper branch）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（lower branch）：b=0 时 La+Lb，b=1 时 La-Lb"""
    u_hat = np.asarray(u_hat)
    if u_hat.ndim == 0:
        return (1 - 2 * u_hat) * La + Lb
    out = np.empty_like(La, dtype=np.float64)
    mask0 = u_hat == 0
    out[mask0] = La[mask0] + Lb[mask0]
    out[~mask0] = La[~mask0] - Lb[~mask0]
    return out


def logdomain_sum(x, y):
    """对数域加法（box-plus 辅助）"""
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation_exact(La, Lb):
    """精确 log-domain f 运算"""
    return logdomain_sum(La + Lb, 0.0) - logdomain_sum(La, Lb)


def bit_reversed_index(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def active_llr_level(i, n):
    """与 py-polar-codes 一致的活跃 LLR 层数"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，半长分解）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def decode(y, f_mask, offset):
        m = len(y)
        if m == 1:
            idx = offset
            if f_mask[0]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if y[0] >= 0 else 1
            return
        h = m // 2
        llr_left = f_operation(y[:h], y[h:])
        decode(llr_left, f_mask[:h], offset)
        llr_right = g_operation(y[:h], y[h:], u_hat[offset:offset + h])
        decode(llr_right, f_mask[h:], offset + h)

    decode(llr, frozen_bits, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits, use_exact_f=False):
    """
    非递归 SC 译码（按比特倒序相位处理，与 polar_encode 配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    f_fn = f_operation_exact if use_exact_f else f_operation

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    upper = f_fn

    def update_llrs(l):
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if top_bit == 0:
                        L[j, s + 1] = L[j, s] + L[j - branch_size, s]
                    else:
                        L[j, s + 1] = L[j, s] - L[j - branch_size, s]

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = bit_reversed_index(i, n)
        update_llrs(l)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)
