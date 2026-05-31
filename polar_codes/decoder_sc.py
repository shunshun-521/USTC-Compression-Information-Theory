"""
极化码 SC（串行抵消）译码器
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(x, n):
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


def _update_llrs(L, B, l, n, N):
    def logdomain_sum(x, y):
        if x > y:
            return x + np.log1p(np.exp(y - x))
        return y + np.log1p(np.exp(x - y))

    def upper_llr(l1, l2):
        return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)

    def lower_llr(l1, l2, b):
        b = int(b) if not np.isnan(b) else 0
        return l1 + l2 if b == 0 else l1 - l2

    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = lower_llr(
                    L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    rev = bit_reversal_permutation(len(llr))
    llr = llr[rev]
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, offset):
        n = len(llr_node)
        if n == 1:
            idx = offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, offset)
        u_left = u_hat[offset : offset + half].copy()
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, offset + half)

    decode_node(llr, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（与 py-polar SCD 一致）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    llr_ch = llr_ch[rev]
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = int(np.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for l in [_bit_reversed(i, n) for i in range(N)]:
        _update_llrs(L, B, l, n, N)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = int(B[l, n])
        _update_bits(B, l, n, N)

    return u_hat


def precompute_sc_indices(N):
    n = int(np.log2(N))
    return (
        [2 ** layer - 1 for layer in range(n + 1)],
        [[s for s in range(n - _active_llr_level(phi, n), n)] for phi in range(N)],
        [[s for s in range(n, n - _active_bit_level(phi, n), -1)] for phi in range(N)],
    )
