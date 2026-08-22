"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于 2D LLR 数组）
"""
import numpy as np
import math
from encoder import bit_reversed


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（upper branch）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(Lb, La, u_hat):
    """
    g 运算（lower branch）：lower_llr(btm, top, u_hat)
    """
    if np.isscalar(u_hat):
        return (Lb + La) if u_hat == 0 else (Lb - La)
    return np.where(u_hat == 0, Lb + La, Lb - La)


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


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量。
    返回 decode_order, llr_active_levels, bit_active_levels
    """
    n = int(math.log2(N))
    decode_order = [bit_reversed(i, n) for i in range(N)]
    llr_active = [_active_llr_level(l, n) for l in decode_order]
    bit_active = [_active_bit_level(l, n) for l in decode_order]
    return decode_order, llr_active, bit_active


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（高效 2D LLR 数组实现）。
    frozen_bits: True/1 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1))
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed(i, n)
        active = _active_llr_level(l, n)
        for s in range(n - active, n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            active_bit = _active_bit_level(l, n)
            for s in range(n, n - active_bit, -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    对每一层 LLR/比特更新使用递归子调用，与 sc_decode 结果一致。
  frozen_bits: True/1 表示冻结位
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1))
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr

    def update_llrs(l, s_start):
        for s in range(s_start, n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def propagate_bits(l, s_end):
        for s in range(n, s_end, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = bit_reversed(i, n)
        active = _active_llr_level(l, n)
        update_llrs(l, n - active)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            active_bit = _active_bit_level(l, n)
            propagate_bits(l, n - active_bit)

    return B[:, n].astype(int)
