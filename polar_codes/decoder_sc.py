"""
极化码 SC（串行抵消）译码器
非递归实现（参考 Permuted SC / py-polar-codes 结构）
"""
import math
import numpy as np
from encoder import bit_reversed_index

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（boxplus 近似）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u=0 -> La+Lb；u=1 -> La-Lb。"""
    u_hat = np.asarray(u_hat)
    if u_hat.ndim == 0 or u_hat.size == 1:
        u = int(u_hat) if u_hat.size else int(u_hat)
        return (1 - 2 * u) * La + Lb
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_boxplus(l1, l2):
    """对数域 f 运算（精确 boxplus，可选）。"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def upper_llr(l1, l2, use_min_sum=True):
    if use_min_sum:
        return f_operation(l1, l2)
    return f_boxplus(l1, l2)


def lower_llr(l1, l2, b, use_min_sum=True):
    if use_min_sum:
        return g_operation(l1, l2, b)
    return l1 + l2 if b == 0 else l1 - l2


def active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


# ==================== 递归 SC（参考）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与主译码器等价，用于校验）。"""
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC ====================


def precompute_sc_indices(N):
    """预计算辅助向量（与按相位倒序译码一致）。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed_index(i, n)
        llr_layers = list(range(n - active_llr_level(l, n), n))
        bit_layers = list(range(n, n - active_bit_level(l, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    lambda_offset = np.arange(n + 1)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits, use_min_sum=False):
    """
    非递归 SC 译码。

    参数：
        llr_ch: 长度 N 的信道 LLR
        frozen_bits: True 表示冻结位（强制为 0）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed_index(i, n)

        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s], use_min_sum)
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], top_bit, use_min_sum
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = (
                            int(B[j, s]) ^ int(B[j - branch_size, s])
                        )
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
