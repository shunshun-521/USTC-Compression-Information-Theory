"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import (
    bit_reversal_permutation,
    map_decoded_to_u_domain,
    transform_frozen_bits,
)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：La 为下分支 LLR，Lb 为上分支 LLR。
    u_hat=0 -> La+Lb；u_hat=1 -> La-Lb
    """
    if u_hat == 0:
        return La + Lb
    return La - Lb


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
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
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _prepare_llr(llr_ch, N):
    """信道 LLR 比特倒序（与编码端 B_N 一致）"""
    return np.asarray(llr_ch, dtype=np.float64)[bit_reversal_permutation(N)]


def _prepare_frozen_mask(frozen_bits, N, u_domain):
    frozen_bits = np.asarray(frozen_bits)
    if u_domain:
        info_idx = np.where(frozen_bits == 0)[0]
        frozen_bits = transform_frozen_bits(frozen_bits, info_idx, N)
    return set(np.where(frozen_bits.astype(int) == 1)[0])


def _sc_decode_prime(llr, frozen_set, n):
    """在 u' 域执行 SCD（Vangala et al. 非递归结构）"""
    N = len(llr)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    decode_order = [_bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N // 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits, u_domain=True):
    """递归 SC 译码（参考实现，调用同一 SCD 核心）"""
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(np.log2(N))
    llr_p = _prepare_llr(llr, N)
    frozen_set = _prepare_frozen_mask(frozen_bits, N, u_domain)
    u_prime = _sc_decode_prime(llr_p, frozen_set, n)
    if u_domain:
        return map_decoded_to_u_domain(u_prime, N)
    return u_prime


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        tmp = phi
        while tmp % 2 == 1:
            llr_layers.append(int(np.log2(tmp & -tmp)))
            tmp //= 2
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        tmp = phi
        while tmp % 2 == 0 and tmp > 0:
            bit_layers.append(int(np.log2(tmp & -tmp)))
            tmp //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits, u_domain=True):
    """非递归 SC 译码主函数"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    llr_p = _prepare_llr(llr_ch, N)
    frozen_set = _prepare_frozen_mask(frozen_bits, N, u_domain)
    u_prime = _sc_decode_prime(llr_p, frozen_set, n)
    if u_domain:
        return map_decoded_to_u_domain(u_prime, N)
    return u_prime
