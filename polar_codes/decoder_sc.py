"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效 SCD 实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（boxplus）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2, min_sum=False):
    if min_sum:
        return float(f_operation(l1, l2))
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, bit, min_sum=False):
    if min_sum:
        return float(g_operation(l1, l2, bit))
    if bit == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，min-sum）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)
    n = int(math.log2(N))

    def decode_node(llr_node, depth, bit_offset):
        m = len(llr_node)
        if m == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return
        half = m // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, depth - 1, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, depth - 1, bit_offset + half)

    decode_node(llr, n, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助信息（解码相位顺序为比特倒序）。
    返回 bit_rev_phases, n
    """
    n = int(math.log2(N))
    phases = [_bit_reversed(i, n) for i in range(N)]
    return phases, n


_SC_CACHE = {}


def sc_decode(llr_ch, frozen_bits, min_sum=True):
    """
    非递归 SC 译码（Permuted SCD，与 ``polar_encode`` 配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    phases, _ = _SC_CACHE[N]

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch

    frozen_set = set(np.where(frozen_bits)[0])

    for l in phases:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s], min_sum)
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j - branch_size, s],
                        L[j, s],
                        int(B[j - branch_size, s + 1]),
                        min_sum,
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N / 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    u_hat = B[:, n].astype(np.int8)
    return u_hat


def sc_decode_verify(llr_ch, frozen_bits):
    """递归与非递归结果一致性（近似）检查。"""
    a = sc_decode(llr_ch, frozen_bits)
    b = sc_decode_recursive(llr_ch, frozen_bits)
    return np.array_equal(a, b)
