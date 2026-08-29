"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效 SCD 实现）
"""
import math
import numpy as np

from encoder import bit_reversed


def f_operation(l1, l2):
    """f 运算（tanh box-plus，数值稳定）。"""
    l1 = np.asarray(l1, dtype=np.float64)
    l2 = np.asarray(l2, dtype=np.float64)
    if l1.shape == ():
        t1 = np.tanh(np.clip(l1 / 2.0, -20.0, 20.0))
        t2 = np.tanh(np.clip(l2 / 2.0, -20.0, 20.0))
        prod = np.clip(t1 * t2, -1.0 + 1e-12, 1.0 - 1e-12)
        return float(2.0 * np.arctanh(prod))
    t1 = np.tanh(np.clip(l1 / 2.0, -20.0, 20.0))
    t2 = np.tanh(np.clip(l2 / 2.0, -20.0, 20.0))
    prod = np.clip(t1 * t2, -1.0 + 1e-12, 1.0 - 1e-12)
    return 2.0 * np.arctanh(prod)


def g_operation(l1, l2, u_hat):
    """g 运算：l1 为下支路，l2 为上支路。"""
    l1 = np.asarray(l1, dtype=np.float64)
    l2 = np.asarray(l2, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    if u_hat.shape == ():
        return l1 + l2 if u_hat == 0 else l1 - l2
    out = np.empty_like(l1, dtype=np.float64)
    mask0 = u_hat == 0
    out[mask0] = l1[mask0] + l2[mask0]
    out[~mask0] = l1[~mask0] - l2[~mask0]
    return out


def f_min_sum(La, Lb, alpha=1.0):
    """min-sum 近似 f 运算（用于 BP）。"""
    return alpha * np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def prepare_channel_llr(llr_ch):
    """信道 LLR 直接用于译码树。"""
    return np.asarray(llr_ch, dtype=np.float64)


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


def _scd_core(llr, frozen_bits, N, n):
    """SCD 核心：返回 B[:, n]。"""
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            branch = block // 2
            for j in range(l, N, block):
                if j % block < branch:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch, s])
                else:
                    top_bit = B[j - branch, s + 1]
                    L[j, s + 1] = g_operation(L[j, s], L[j - branch, s], top_bit)

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block = 1 << s
            branch = block // 2
            for j in range(l, -1, -block):
                if j % block >= branch:
                    B[j - branch, s - 1] = B[j, s] ^ B[j - branch, s]
                    B[j, s - 1] = B[j, s]

    for phi_nat in range(N):
        l = bit_reversed(phi_nat, n)
        update_llrs(l)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 SCD 共享冻结位语义）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    return _scd_core(np.asarray(llr, dtype=np.float64), frozen_bits, N, n)


def precompute_sc_indices(N):
    """预计算 SCL 使用的层索引。"""
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for phi in range(N):
        l = bit_reversed(phi, n)
        for s in range(n - _active_llr_level(l, n), n):
            llr_layer_vec[phi].append(s)
        for s in range(n, n - _active_bit_level(l, n), -1):
            bit_layer_vec[phi].append(s)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr = prepare_channel_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    return _scd_core(llr, frozen_bits, N, n)
