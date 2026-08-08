"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math
from encoder import bit_reversed


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """从最高位起统计连续 0 的个数"""
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
    """从最高位起统计连续 1 的个数"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _combine_subtrees(left_bits, right_bits):
    left_bits = np.asarray(left_bits, dtype=int)
    right_bits = np.asarray(right_bits, dtype=int)
    return np.concatenate([(left_bits + right_bits) % 2, right_bits])


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    frozen_bits: True 表示冻结位
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    bit_counter = [0]

    def decode_node(y):
        if len(y) == 1:
            idx = bit_counter[0]
            if frozen_bits[idx]:
                bit = 0
            else:
                bit = 0 if y[0] >= 0 else 1
            bit_counter[0] += 1
            return np.array([bit], dtype=int)

        half = len(y) // 2
        y_left = y[:half]
        y_right = y[half:]
        bits_left = decode_node(f_operation(y_left, y_right))
        bits_right = decode_node(g_operation(y_left, y_right, bits_left))
        return _combine_subtrees(bits_left, bits_right)

    return decode_node(llr)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        temp = phi
        while temp % 2 == 1:
            layers_llr.append(int(math.log2(temp & -temp)) - 1)
            temp >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 1:
            temp = phi
            while temp % 2 == 1:
                layers_bit.append(int(math.log2(temp & -temp)) - 1)
                temp >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    frozen_bits: 1 表示冻结位，0 表示信息位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits == 1)[0])

    def update_llrs(phi):
        for s in range(n - _active_llr_level(phi, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(phi, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )

    def update_bits(phi):
        if phi < N // 2:
            return
        for s in range(n, n - _active_bit_level(phi, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(phi, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        phi = bit_reversed(i, n)
        update_llrs(phi)
        if phi in frozen_set:
            u_hat[phi] = 0
            B[phi, n] = 0
        else:
            u_hat[phi] = 0 if L[phi, n] >= 0 else 1
            B[phi, n] = u_hat[phi]
        update_bits(phi)

    return u_hat
