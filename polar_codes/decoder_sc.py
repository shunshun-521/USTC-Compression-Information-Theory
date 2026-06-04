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
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(a, b):
    if a == b:
        return a + np.log(2.0)
    if a > b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


def _upper_llr_exact(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr_exact(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


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


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（与标准 SCD 层索引对应）
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi in range(N):
        start = n - _active_llr_level(_bit_reversed(phi, n), n)
        llr_layer_vec[phi] = list(range(start, n))
        br_phi = _bit_reversed(phi, n)
        if br_phi < N / 2:
            bit_layer_vec[phi] = []
        else:
            start_bit = n - _active_bit_level(br_phi, n)
            bit_layer_vec[phi] = list(range(n, start_bit, -1))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，使用精确 box-plus）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, frozen_node, offset):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                u_hat[offset] = 0
            else:
                u_hat[offset] = 0 if llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_left = np.array(
            [_upper_llr_exact(llr_node[i], llr_node[i + half]) for i in range(half)]
        )
        decode_node(llr_left, frozen_node[:half], offset)
        u_left = u_hat[offset : offset + half]
        llr_right = np.array(
            [
                _lower_llr_exact(llr_node[i], llr_node[i + half], u_left[i])
                for i in range(half)
            ]
        )
        decode_node(llr_right, frozen_node[half:], offset + half)

    decode_node(llr, frozen_bits, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（因子图矩阵遍历，精确 box-plus）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch[br]

    for phase in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(phase, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(phase, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_exact(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr_exact(
                        L[j, s],
                        L[j - branch_size, s],
                        int(B[j - branch_size, s + 1]),
                    )

        if phase in frozen_set:
            B[phase, n] = 0
        else:
            B[phase, n] = 0 if L[phase, n] >= 0 else 1

        if phase >= N // 2:
            for s in range(n, n - _active_bit_level(phase, n), -1):
                block_size = 1 << s
                branch_size = block_size >> 1
                for j in range(phase, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
