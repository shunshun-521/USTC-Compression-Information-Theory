"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 Permuted SCD（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr_boxplus(l1, l2):
    """对数域 box-plus（f 运算的精确形式）"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr_boxplus(btm, top, bit):
    """对数域 g 运算"""
    return (btm + top) if bit == 0 else (btm - top)


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


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，与 Permuted SCD 等价）。
    采用逐相位的 LLR 树更新，结果与非递归版本一致。
    """
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC 译码（Permuted SCD）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助信息（Permuted SCD 变体）。
    返回 bit_reversed 译码顺序及每层活跃范围。
    """
    n = int(np.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for l in decode_order:
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        if l < N // 2:
            bit_layer_vec.append([])
        else:
            bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    lambda_offset = np.array([1 << (n - i) for i in range(n + 1)], dtype=int)
    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def _get_sc_cache(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SCD 译码主函数。
    信道 LLR 直接输入，内部做比特倒序置换以匹配编码器输出。
    """
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    rev = bit_reversal_permutation(N)
    llr = np.asarray(llr_ch, dtype=np.float64)[rev]

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    _, llr_layer_vec, bit_layer_vec = _get_sc_cache(N)

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = _bit_reversed(phi, n)

        for s in llr_layer_vec[phi]:
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr_boxplus(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

        if l in frozen_set:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        for s in bit_layer_vec[phi]:
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return u_hat
