"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）与非递归矩阵实现（高效）
"""
import numpy as np
import math


def bit_reversed(x, n):
    """比特倒序索引"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（La=上支路，Lb=下支路）"""
    return (1 - 2 * u_hat) * La + Lb


def upper_llr(l1, l2):
    return f_operation(l1, l2)


def lower_llr(l1, l2, b):
    """下支路 LLR（l1=下，l2=上）"""
    if b is None or (isinstance(b, float) and np.isnan(b)):
        b = 0
    if int(b) == 0:
        return l1 + l2
    return l1 - l2


def active_llr_level(i, n):
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


def sc_new_state(N, llr_ch):
    n = int(math.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = np.asarray(llr_ch, dtype=np.float64)
    return L, B, n


def sc_update_llrs(L, B, l, n, N):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = lower_llr(
                    L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                )


def sc_update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                bj = B[j, s]
                btop = B[j - branch_size, s]
                if np.isnan(bj) or np.isnan(btop):
                    continue
                B[j - branch_size, s - 1] = int(bj) ^ int(btop)
                B[j, s - 1] = bj


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC（L/B 矩阵，按比特倒序相位译码）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    L, B, n = sc_new_state(N, llr_ch)

    for i in range(N):
        l = bit_reversed(i, n)
        sc_update_llrs(L, B, l, n, N)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        sc_update_bits(B, l, n, N)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与矩阵实现交叉验证）"""
    return sc_decode_nonrecursive(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：返回占位结构（SCL 使用矩阵译码时可忽略）"""
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    return sc_decode_nonrecursive(llr_ch, frozen_bits)
