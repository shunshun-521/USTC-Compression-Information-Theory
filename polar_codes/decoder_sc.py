"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：等价于 b + a (u=0) 或 b - a (u=1)"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，与 dit.coding.polar 结构一致）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    decisions = np.zeros(N, dtype=int)
    indices = list(range(N))

    def _sc(llrs, idx_list):
        if len(idx_list) == 1:
            i = idx_list[0]
            if frozen_bits[i]:
                decisions[i] = 0
            else:
                decisions[i] = 0 if llrs[0] >= 0 else 1
            return [decisions[i]]

        half = len(idx_list) // 2
        left_llr = [
            f_operation(llrs[i], llrs[i + half]) for i in range(half)
        ]
        enc_left = _sc(left_llr, idx_list[:half])
        right_llr = [
            g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
        ]
        enc_right = _sc(right_llr, idx_list[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    _sc(llr, indices)
    return decisions


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量（自然序相位）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = 0
        while l < n and ((phi >> l) & 1):
            l += 1
        llr_layer_vec.append(list(range(l, n)))
        bit_layer_vec.append(list(range(l)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _active_llr_level(i, n):
    """从 MSB 起找到第一个 1 的位置（与 polarcodes 一致）。"""
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
    """从 MSB 起找到第一个 0 的位置。"""
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
    return int(format(i, f"0{n}b")[::-1], 2)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """
    非递归 SC 译码（参考 polarcodes.SCD 的 LLR/比特更新结构）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    def upper_llr(a, b):
        return f_operation(a, b)

    def lower_llr(btm, top, top_bit):
        return g_operation(top, btm, top_bit)

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top_llr = L[j, s]
                    btm_llr = L[j + branch_size, s]
                    L[j, s + 1] = upper_llr(top_llr, btm_llr)
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = lower_llr(btm_llr, top_llr, top_bit)

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for l_nat in [_bit_reversed(i, n) for i in range(N)]:
        update_llrs(l_nat)
        if frozen_bits[l_nat]:
            B[l_nat, n] = 0
        else:
            B[l_nat, n] = 0 if L[l_nat, n] >= 0 else 1
        u_hat[l_nat] = B[l_nat, n]
        update_bits(l_nat)

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """默认使用高效非递归 SC 译码。"""
    return sc_decode_nonrecursive(llr_ch, frozen_bits)
