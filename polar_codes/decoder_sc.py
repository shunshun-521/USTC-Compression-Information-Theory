"""
极化码 SC（串行抵消）译码器
实现参考标准非递归 SC（与蝶形编码 stride-first 配套）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（upper branch）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（lower branch）"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """LLR 更新起始层（首个为 1 的比特位置，MSB 起）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    """比特回传起始层（首个为 0 的比特位置）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n, N):
    """更新比特 l 的 LLR 树。"""
    start = n - _active_llr_level(l, n)
    for s in range(start, n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], top_bit
                )


def _update_bits(B, l, n, N):
    """比特回传。"""
    if l < N // 2:
        return
    end_s = n - _active_bit_level(l, n)
    for s in range(n, end_s, -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 等价）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << (i - 1))
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    按比特倒序索引顺序译码；L[:,0] 为信道 LLR。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)
    rev = bit_reversal_permutation(N)
    decode_order = [int(rev[i]) for i in range(N)]

    for l in decode_order:
        _update_llrs(L, B, l, n, N)
        if frozen_bits[l]:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]
        _update_bits(B, l, n, N)

    return u_hat
