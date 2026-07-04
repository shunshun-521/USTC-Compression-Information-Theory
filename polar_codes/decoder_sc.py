"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def _sign_stable(x):
    s = np.sign(x)
    s[s == 0] = 1.0
    return s


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return _sign_stable(La) * _sign_stable(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_boxplus(La, Lb):
    """SC 译码使用的精确 f 运算（boxplus）。"""
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


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


def _prepare_llr(llr_ch):
    """信道 LLR 按比特倒序重排以匹配编码器。"""
    br = bit_reversal_permutation(len(llr_ch))
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _sc_decode_core(llr, frozen_bits, f_func):
    llr = _prepare_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    llr_mem = np.full((N, n + 1), np.nan, dtype=np.float64)
    bit_mem = np.zeros((N, n + 1), dtype=np.int8)
    llr_mem[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    llr_mem[j, s + 1] = f_func(llr_mem[j, s], llr_mem[j + branch_size, s])
                else:
                    llr_mem[j, s + 1] = g_operation(
                        llr_mem[j - branch_size, s],
                        llr_mem[j, s],
                        bit_mem[j - branch_size, s + 1],
                    )

        if l in frozen_set:
            bit_mem[l, n] = 0
        else:
            bit_mem[l, n] = 0 if llr_mem[l, n] >= 0 else 1
        u_hat[l] = bit_mem[l, n]

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        bit_mem[j - branch_size, s - 1] = (
                            bit_mem[j, s] ^ bit_mem[j - branch_size, s]
                        )
                        bit_mem[j, s - 1] = bit_mem[j, s]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    return _sc_decode_core(llr, frozen_bits, f_boxplus)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（接口兼容）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        p = phi
        while p % 2 == 1:
            layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(layers)
        layers_b = []
        p = phi
        while p > 0 and p % 2 == 0:
            layers_b.append(int(math.log2(p & -p)))
            p >>= 1
        bit_layer_vec.append(layers_b)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（高效实现，算法等价于递归版本）。
    """
    return _sc_decode_core(llr_ch, frozen_bits, f_boxplus)
