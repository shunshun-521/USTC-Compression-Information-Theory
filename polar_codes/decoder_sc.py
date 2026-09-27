"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（与 polarcodes lower_llr 在 min-sum 下等价）"""
    return (1 - 2 * u_hat) * La + Lb


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


def _bit_reversed(i, n):
    rev = 0
    v = i
    for _ in range(n):
        rev = (rev << 1) | (v & 1)
        v >>= 1
    return rev


def _map_channel_llrs(llr_ch):
    """x[i]=v[rev(i)] 时，将信道 LLR 映射到蝶形域索引"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=np.int64)
    inv[rev] = np.arange(N)
    mapped = np.zeros(N, dtype=np.float64)
    mapped[:] = llr_ch[inv]
    return mapped


def _scd_core(llr_mapped, frozen_bits):
    """非递归 SCD（参考 polarcodes 结构，f 用 min-sum）"""
    N = len(llr_mapped)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_mapped

    def upper_llr(l1, l2):
        return float(f_operation(l1, l2))

    def lower_llr(l1, l2, b):
        if b == 0:
            return l1 + l2
        return l1 - l2

    def update_llrs(l):
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

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = _bit_reversed(i, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(np.int8)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr_mapped = _map_channel_llrs(llr_ch)
    return _scd_core(llr_mapped, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 共用同一 LLR 映射与 SCD 核心）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（供 SCL 使用）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(n - 1, start - 1, -1)))
        bit_layer_vec.append(list(range(_active_bit_level(l, n))))
    return lambda_offset, llr_layer_vec, bit_layer_vec
