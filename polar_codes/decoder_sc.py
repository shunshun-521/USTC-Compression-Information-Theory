"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation, bit_reversed_index


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    abs_a = np.abs(La)
    abs_b = np.abs(Lb)
    sign_a = np.where(abs_a < 1e-15, 1.0, np.sign(La))
    sign_b = np.where(abs_b < 1e-15, 1.0, np.sign(Lb))
    combined = sign_a * sign_b * np.minimum(abs_a, abs_b)
    return np.where(abs_a < 1e-15, Lb, np.where(abs_b < 1e-15, La, combined))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


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


def _align_channel_llrs(llr_ch):
    """将信道 LLR 对齐到蝶形树自然顺序（与编码端比特倒序对应）。"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = _align_channel_llrs(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                return np.array([0], dtype=int)
            return np.array([0 if llr_node[0] >= 0 else 1], dtype=int)

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left = decode_node(llr_left, frozen_node[:half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        u_right = decode_node(llr_right, frozen_node[half:])
        return np.concatenate([u_left, u_right])

    return decode_node(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助索引。"""
    n = int(np.log2(N))
    decode_order = [bit_reversed_index(i, n) for i in range(N)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi, l in enumerate(decode_order):
        start = n - _active_llr_level(l, n)
        llr_layer_vec[phi] = list(range(start, n))
        if l < N // 2:
            bit_layer_vec[phi] = []
        else:
            start_bit = n - _active_bit_level(l, n)
            bit_layer_vec[phi] = list(range(n, start_bit, -1))

    return decode_order, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码。"""
    llr_ch = _align_channel_llrs(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    decode_order, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi, l in enumerate(decode_order):
        for s in llr_layer_vec[phi]:
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        top_bit,
                    )

        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        for s in bit_layer_vec[phi]:
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    return u_hat
