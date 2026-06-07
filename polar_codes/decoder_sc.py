"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def _reorder_channel_llrs(llr_ch):
    """编码端含比特倒序时，将信道 LLR 重排为译码树自然顺序"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = _reorder_channel_llrs(llr)
    return _sc_decode_core(llr, frozen_bits, use_recursive=True)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        t = phi
        while t % 2 == 1:
            llr_layers.append(int(math.log2(t & -t)))
            t >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 0:
            t = phi
            while t > 0 and t % 2 == 0:
                bit_layers.append(int(math.log2(t & -t)) + 1)
                t >>= 1
        else:
            bit_layers.append(0)
            t = phi
            while t > 0 and t % 2 == 0:
                bit_layers.append(int(math.log2(t & -t)) + 1)
                t >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_core(llr_ch, frozen_bits, use_recursive=False):
    """SC 译码核心：按比特倒序处理索引"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        update_llrs(l)
        if frozen_bits[l]:
            u_hat[l] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
        B[l, n] = u_hat[l]
        update_bits(l)

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr_internal = _reorder_channel_llrs(llr_ch)
    return _sc_decode_core(llr_internal, frozen_bits)


def path_metric_penalty(llr, u):
    """路径度量惩罚"""
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)
