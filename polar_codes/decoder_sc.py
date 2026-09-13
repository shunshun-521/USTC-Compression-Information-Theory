"""
极化码 SC（串行抵消）译码器
提供 PSCD（置换 SC）高效实现与递归参考实现
"""
import math

import numpy as np

from encoder import bit_reversed


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（box-plus 近似）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sign_a = np.where(La >= 0, 1.0, -1.0)
    sign_b = np.where(Lb >= 0, 1.0, -1.0)
    return sign_a * sign_b * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if np.isscalar(x) and np.isscalar(y):
        if x > y:
            return x + np.log1p(np.exp(y - x))
        return y + np.log1p(np.exp(x - y))
    return np.where(
        x > y,
        x + np.log1p(np.exp(y - x)),
        y + np.log1p(np.exp(x - y)),
    )


def _box_plus(l1, l2):
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _upper_llr(l1, l2):
    return _box_plus(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


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


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, frozen_segment):
        n = len(llr_node)
        if n == 1:
            if frozen_segment[0]:
                return np.array([0], dtype=int)
            return np.array([0 if llr_node[0] >= 0 else 1], dtype=int)

        half = n // 2
        llr_left = np.array([_upper_llr(a, b) for a, b in zip(llr_node[:half], llr_node[half:])])
        u_left = decode_node(llr_left, frozen_segment[:half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        u_right = decode_node(llr_right, frozen_segment[half:])
        return np.concatenate([u_left, u_right])

    return decode_node(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算 PSCD 译码辅助信息（兼容接口）。"""
    n = int(math.log2(N))
    decode_order = [bit_reversed(i, n) for i in range(N)]
    return decode_order, n


def sc_decode(llr_ch, frozen_bits):
    """
    PSCD 非递归 SC 译码主函数。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for l in [bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s],
                        L[j - branch_size, s],
                        int(B[j - branch_size, s + 1]),
                    )

        if l in frozen_set:
            u_hat[l] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
        B[l, n] = u_hat[l]

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return u_hat
