"""
极化码 SC（串行抵消）译码器
非递归 SCD（参考 Vangala et al. / mcba1n polar-codes）
"""
import math
import numpy as np

from decoder_utils import (
    active_bit_level,
    active_llr_level,
    bit_reversed_index,
    frozen_indices_from_mask,
    lower_llr,
    upper_llr,
)


def f_operation(La, Lb):
    """向量化的 upper_llr（box-plus）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.vectorize(upper_llr)(La, Lb)


def f_operation_min_sum(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """与 lower_llr 一致：La 为上支路，Lb 为下支路"""
    u_hat = np.asarray(u_hat)
    if u_hat.ndim == 0 or u_hat.size == 1:
        return lower_llr(float(La), float(Lb), int(u_hat))
    out = np.empty_like(La, dtype=np.float64)
    for i in range(len(La)):
        out[i] = lower_llr(La[i], Lb[i], int(u_hat[i]))
    return out


def _sc_decode_core(llr_ch, frozen_set, n, N):
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for phase in range(N):
        l = bit_reversed_index(phase, n)
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], top_bit
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = frozen_indices_from_mask(frozen_bits)
    return _sc_decode_core(llr_ch, frozen_set, n, N)


def sc_decode_recursive(llr, frozen_bits):
    """递归接口：与 sc_decode 等价（保留 API）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """兼容 SCL：返回占位结构（SCL 使用独立实现）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]
