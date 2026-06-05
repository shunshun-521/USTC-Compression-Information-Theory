"""
极化码 SC（串行抵消）译码器
主实现：硬判决 + 逆蝶形（与 B_N F^{⊗n} 编码一致）
备选：SSC 软译码 / py-polar SCD 软译码
"""
import importlib.util
import math
import sys

import numpy as np

from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(top, btm, b):
    b = int(b)
    return top + btm if b == 0 else top - btm


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


def inverse_polar_hard(x_bits):
    N = len(x_bits)
    br = bit_reversal_permutation(N)
    u = np.array(x_bits, dtype=np.int8)[br].copy()
    step = N // 2
    while step >= 1:
        for left in range(0, N, 2 * step):
            u[left : left + step] ^= u[left + step : left + 2 * step]
        step //= 2
    return u.astype(int)


def sc_decode_peel(llr_ch, frozen_bits):
    x_hard = (llr_ch < 0).astype(int)
    u_hat = inverse_polar_hard(x_hard)
    u_hat[np.asarray(frozen_bits, dtype=bool)] = 0
    return u_hat


# ==================== SSC 软 SC（v 域，比特倒序译码顺序）====================

_SC_CACHE = {}


def precompute_sc_indices(N):
    if N in _SC_CACHE:
        return _SC_CACHE[N]
    n = int(math.log2(N))
    order = [int(f"{i:0{n}b}"[::-1], 2) for i in range(N)]
    _SC_CACHE[N] = (n, order)
    return _SC_CACHE[N]


def sc_decode_ssc(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n, order = precompute_sc_indices(N)
    br = bit_reversal_permutation(N)
    llr_v = llr_ch[br]
    frozen_v = frozen_bits[br]
    frozen_set = set(np.where(frozen_v)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_v

    for l in order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2**s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = (
                            int(B[j, s]) ^ int(B[j - branch_size, s])
                        )
                        B[j, s - 1] = B[j, s]

    u_v = B[:, n].astype(int)
    u_hat = np.zeros(N, dtype=int)
    u_hat[br] = u_v
    return u_hat


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 min-sum SC（v 域）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr_v = llr_ch[br]
    frozen_v = frozen_bits[br]
    u_v = np.zeros(N, dtype=int)

    def dec(node, off):
        n = len(node)
        if n == 1:
            i = off
            u_v[i] = 0 if frozen_v[i] or node[0] >= 0 else 1
            return
        h = n // 2
        dec(f_operation(node[:h], node[h:]), off)
        dec(g_operation(node[:h], node[h:], u_v[off : off + h]), off + h)

    dec(llr_v, 0)
    u_hat = np.zeros(N, dtype=int)
    u_hat[br] = u_v
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口"""
    return sc_decode_peel(llr_ch, frozen_bits)
