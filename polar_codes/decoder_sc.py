"""
极化码 SC（串行抵消）译码器
采用 Permuted SCD（PSCD）实现，与蝶形编码 + 比特倒序一致
"""
import math
import numpy as np


def bit_reversed(i, n):
    """将 i 的 n 位二进制表示做比特倒序"""
    return int(f"{i:0{n}b}"[::-1], 2)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def active_llr_level(i, n):
    """二进制表示中第一个 1 的位置（从高位计）"""
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
    """二进制表示中第一个 0 的位置（从高位计）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n, N):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                top_llr = L[j, s]
                btm_llr = L[j + branch_size, s]
                L[j, s + 1] = f_operation(top_llr, btm_llr)
            else:
                btm_llr = L[j, s]
                top_llr = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 PSCD 译码。
    frozen_bits: 1 表示冻结位，0 表示信息位（自然顺序 u 索引）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed(i, n)
        _update_llrs(L, B, l, n, N)
        if frozen_bits[i]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
        u_hat[i] = int(B[bit_reversed(i, n), n])
    return u_hat


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 参考实现（调用 PSCD）"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """保留接口兼容性"""
    n = int(math.log2(N))
    lambda_offset = [(1 << l) - 1 for l in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(active_llr_level(bit_reversed(phi, n), n))))
        bit_layer_vec.append(list(range(active_bit_level(bit_reversed(phi, n), n))))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def reorder_channel_llr(llr_ch):
    """PSCD 直接使用自然顺序信道 LLR，无需倒序"""
    return np.asarray(llr_ch, dtype=np.float64)
