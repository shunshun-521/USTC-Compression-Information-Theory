"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(x, n):
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


def _reorder_llrs(llr_ch):
    br = bit_reversal_permutation(len(llr_ch))
    return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（Sionna/Hashemi 风格，含部分和回传）。"""
    llr = _reorder_llrs(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, frozen_node):
        n_len = len(llr_node)
        if n_len == 1:
            if frozen_node[0]:
                bit = 0.0
            else:
                bit = 0.0 if llr_node[0] >= 0 else 1.0
            u_arr = np.array([bit], dtype=np.float64)
            return np.array([int(bit)]), u_arr

        half = n_len // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left, u_left_up = decode_node(llr_left, frozen_node[:half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
        u_right, u_right_up = decode_node(llr_right, frozen_node[half:])
        u_hat = np.concatenate([u_left, u_right])
        u_left_up_int = (u_left_up.astype(int) ^ u_right_up.astype(int)).astype(np.float64)
        u_hat_up = np.concatenate([u_left_up_int, u_right_up])
        return u_hat, u_hat_up

    u_hat, _ = decode_node(llr, frozen_bits)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        while psi % 2 == 1:
            llr_layers.append(int(math.log2(psi & -psi)))
            psi //= 2
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        for layer in range(n):
            if (phi >> layer) & 1:
                break
            bit_layers.append(layer)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（Permuted SCD，与 mcba1n 一致）。"""
    llr = _reorder_llrs(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n_len = len(llr)
    n = int(math.log2(n_len))
    frozen_set = set(np.where(frozen_bits)[0])

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(n_len):
        l_idx = _bit_reversed(phi, n)
        llr_layer_vec.append(range(n - _active_llr_level(l_idx, n), n))
        if l_idx >= n_len / 2:
            bit_layer_vec.append(range(n, n - _active_bit_level(l_idx, n), -1))
        else:
            bit_layer_vec.append([])

    l_mat = np.full((n_len, n + 1), np.nan, dtype=np.float64)
    c_mat = np.full((n_len, n + 1), np.nan, dtype=np.float64)
    l_mat[:, 0] = llr
    u_hat = np.zeros(n_len, dtype=int)

    for phi in range(n_len):
        l_idx = _bit_reversed(phi, n)
        for s in llr_layer_vec[phi]:
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l_idx, n_len, block_size):
                if j % block_size < branch_size:
                    l_mat[j, s + 1] = f_operation(l_mat[j, s], l_mat[j + branch_size, s])
                else:
                    l_mat[j, s + 1] = g_operation(
                        l_mat[j - branch_size, s],
                        l_mat[j, s],
                        c_mat[j - branch_size, s + 1],
                    )

        if l_idx in frozen_set:
            c_mat[l_idx, n] = 0
        else:
            c_mat[l_idx, n] = 0 if l_mat[l_idx, n] >= 0 else 1
        u_hat[l_idx] = int(c_mat[l_idx, n])

        if l_idx < n_len / 2:
            continue

        for s in bit_layer_vec[phi]:
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    c_mat[j - branch_size, s - 1] = int(c_mat[j, s]) ^ int(c_mat[j - branch_size, s])
                    c_mat[j, s - 1] = c_mat[j, s]

    return u_hat
