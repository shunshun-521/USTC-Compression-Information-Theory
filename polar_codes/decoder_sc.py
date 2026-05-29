"""
极化码 SC（串行抵消）译码器
非递归节点更新（与 polar_encode 配套）
"""
import numpy as np
import math
from decoder_utils import (
    upper_llr,
    lower_llr,
    hard_decision,
    active_llr_level,
    active_bit_level,
    bit_reversed,
)


def f_operation(La, Lb):
    """min-sum 近似 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（La 上支路，Lb 下支路）"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(int(i) for i in np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for phase in [bit_reversed(i, n) for i in range(N)]:
        for s in range(n - active_llr_level(phase, n), n):
            block_size = int(2 ** (s + 1))
            branch_size = block_size // 2
            for j in range(phase, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = lower_llr(btm_llr, top_llr, top_bit)

        if phase in frozen_set:
            B[phase, n] = 0
        else:
            B[phase, n] = hard_decision(L[phase, n])

        if phase < N // 2:
            continue

        for s in range(n, n - active_bit_level(phase, n), -1):
            block_size = int(2 ** s)
            branch_size = block_size // 2
            for j in range(phase, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    return list(range(n + 1)), [[] for _ in range(N)], [[] for _ in range(N)]
