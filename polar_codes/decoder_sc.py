"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(i, n):
  return int(format(i, f"0{n}b")[::-1], 2)


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


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, depth, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return u_hat[idx]

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left = np.zeros(half, dtype=int)
        for i in range(half):
            u_left[i] = decode_node(llr_left[i : i + 1], depth - 1, bit_offset + i)
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i : i + 1], depth - 1, bit_offset + half + i)
        return None

    decode_node(llr, int(math.log2(N)), 0)
    return u_hat


def sc_decode_recursive_channel(llr_ch, frozen_bits):
    """递归 SC 译码（自动完成信道 LLR 重排）。"""
    llr = _reorder_channel_llr(llr_ch, len(llr_ch))
    return sc_decode_recursive(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed(phi, n)
        start_llr = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start_llr, n)))

        if l < N // 2:
            bit_layer_vec.append([])
        else:
            start_bit = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, start_bit, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs(l, n, L, B):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, len(L), block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(l, n, B):
    if l < len(B) // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _reorder_channel_llr(llr_ch, N):
    """将信道 LLR 重排为与蝶形编码一致的顺序。"""
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    llr_ch = _reorder_channel_llr(llr_ch, len(llr_ch))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for phi in range(N):
        l = _bit_reversed(phi, n)
        _update_llrs(l, n, L, B)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        _update_bits(l, n, B)

    return B[:, n].astype(int)
