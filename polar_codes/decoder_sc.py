"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Vangala PSC 风格）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _g_scalar(a, b, u):
    return b - a if u else b + a


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（与 dit.polar 一致：g 使用子树编码比特）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    decisions = np.zeros(N, dtype=int)

    def _sc(llrs, indices):
        if len(indices) == 1:
            i = indices[0]
            if frozen_bits[i]:
                decisions[i] = 0
            else:
                decisions[i] = 0 if llrs[0] >= 0 else 1
            return [decisions[i]]

        half = len(indices) // 2
        left_llr = [
            float(f_operation(llrs[i], llrs[i + half])) for i in range(half)
        ]
        enc_left = _sc(left_llr, indices[:half])
        right_llr = [
            _g_scalar(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
        ]
        enc_right = _sc(right_llr, indices[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    _sc(llr.tolist(), list(range(N)))
    return decisions


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


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
    n = int(math.log2(N))
    phases = [_bit_reversed(i, n) for i in range(N)]
    return phases, None, None


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 PSC 译码（Vangala 2014 风格，min-sum，小码长验证用）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch
    frozen_set = set(np.where(frozen_bits)[0])

    for phi in range(N):
        l = _bit_reversed(phi, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = float(
                        f_operation(L[j, s], L[j + branch_size, s])
                    )
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if top_bit == 0:
                        L[j, s + 1] = L[j - branch_size, s] + L[j, s]
                    else:
                        L[j, s + 1] = L[j - branch_size, s] - L[j, s]

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N / 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    u_hat = np.zeros(N, dtype=int)
    for phi in range(N):
        l = _bit_reversed(phi, n)
        u_hat[phi] = B[l, n]
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（默认使用经验证的递归实现）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
