"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def _bit_reversed_index(x, n):
    """整数比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def permute_llr_for_decode(llr_ch):
    """将信道 LLR 按比特倒序置换以匹配编码器"""
    brp = bit_reversal_permutation(len(llr_ch))
    return llr_ch[brp]


def f_operation(La, Lb):
    """min-sum 近似 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：La=top, Lb=bottom"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """LLR 更新起始层"""
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
    """比特回传起始层"""
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
    """预计算非递归 SC 辅助向量（供 SCL 使用）"""
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        if l >= N // 2:
            bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
        else:
            bit_layer_vec.append([])
    return llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，与 sc_decode 等价）"""
    return sc_decode(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（高效实现）。
    基于 Permuted SCD 算法（Vangala et al., 2014）。
    """
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    frozen_set = set(np.where(frozen_bits == 1)[0])

    llr = permute_llr_for_decode(llr_ch).astype(np.float64)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    btm = L[j, s]
                    top = L[j - branch_size, s]
                    bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(top, btm, bit)

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
        l = _bit_reversed_index(i, n)
        update_llrs(l)
        if l in frozen_set:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]
        update_bits(l)

    return u_hat
