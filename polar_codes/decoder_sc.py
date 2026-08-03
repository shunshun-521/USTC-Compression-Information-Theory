"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """min-sum g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    """对数域 f 运算（精确）。"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    """对数域 g 运算（精确）。"""
    return l1 + l2 if b == 0 else l1 - l2


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
    """递归 SC 译码（参考实现，对数域）。"""
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode(llrs, bit_idx):
        m = len(llrs)
        if m == 1:
            if frozen_bits[bit_idx]:
                u_hat[bit_idx] = 0
            else:
                u_hat[bit_idx] = 0 if llrs[0] >= 0 else 1
            return

        half = m // 2
        llr_left = np.array([_upper_llr(llrs[i], llrs[i + half]) for i in range(half)])
        decode(llr_left, bit_idx)
        llr_right = np.array([
            _lower_llr(llrs[i + half], llrs[i], u_hat[bit_idx + i])
            for i in range(half)
        ])
        decode(llr_right, bit_idx + half)

    decode(llr.copy(), 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（对数域，比特倒序相位处理）。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch.copy()

    br = bit_reversal_permutation(N)

    for idx in range(N):
        l = br[idx]

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
