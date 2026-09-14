"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于 PSC 算法）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（upper_llr）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（lower_llr）"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    """i 的二进制表示中第一个 1 的位置（从高位计）"""
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
    """i 的二进制表示中第一个 0 的位置（从高位计）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def build_decoder_frozen_bits(N, info_indices):
    """自然序冻结位数组（1=冻结）"""
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_indices] = 0
    return frozen_bits


def to_natural_order(decoded, N):
    """恒等映射（译码器已输出自然序）"""
    return decoded


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（信道序 LLR，frozen_bits 为自然序）"""
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int))[0])
    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr_ch[br]

    for i in range(N):
        l = _bit_reversed(i, n)
        _update_llrs_psc(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits_psc(B, l, n)

    return B[:, n].astype(int)


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """预计算 PSC 风格的译码顺序与层索引"""
    n = int(math.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for l in decode_order:
        start_s = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start_s, n)))
        bit_start = n - _active_bit_level(l, n) + 1
        bit_layer_vec.append(list(range(n, bit_start - 1, -1)) if l >= N // 2 else [])
    return decode_order, llr_layer_vec, bit_layer_vec


def _update_llrs_psc(L, B, l, n):
    """PSC 风格 LLR 更新"""
    N = L.shape[0]
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits_psc(B, l, n):
    """PSC 风格比特回传"""
    N = B.shape[0]
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]


def sc_decode_channel(llr_ch, frozen_bits):
    """非递归 SC 译码（frozen_bits 自然序，1=冻结）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr_ch[br]

    decode_order, _, _ = precompute_sc_indices(N)
    for l in decode_order:
        _update_llrs_psc(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits_psc(B, l, n)

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（自然序 frozen_bits，返回自然序 u_hat）"""
    return sc_decode_channel(llr_ch, frozen_bits)
