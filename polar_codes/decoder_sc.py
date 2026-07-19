"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


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
    return (1.0 - 2.0 * np.asarray(u_hat, dtype=np.float64)) * La + Lb


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
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
  信道 LLR 在入口做比特倒序以匹配编码器输出顺序。
    """
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    llr = llr[br]
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, depth, bit_offset):
        m = len(llr_node)
        if m == 1:
            idx = bit_offset
            if idx in frozen_set:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return u_hat[idx]

        half = m // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left = np.zeros(half, dtype=int)
        for i in range(half):
            u_left[i] = decode_node(llr_left[i : i + 1], depth - 1, bit_offset + i)
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i : i + 1], depth - 1, bit_offset + half + i)
        return None

    decode_node(llr, n, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回比特倒序处理顺序及每层活跃标记。
    """
    n = int(math.log2(N))
    lambda_offset = np.array([1 << i for i in range(n + 1)], dtype=int)
    process_order = [_bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []

    for l in process_order:
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        if l < N // 2:
            bit_layer_vec.append([])
        else:
            bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec, process_order


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Permuted SC，与蝶形+比特倒序编码配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    llr_ch = llr_ch[br]

    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    for i in range(N):
        l = _bit_reversed(i, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            half = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < half:
                    L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - half, s], L[j, s], B[j - half, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N // 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            half = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= half:
                    B[j - half, s - 1] = B[j, s] ^ B[j - half, s]
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
