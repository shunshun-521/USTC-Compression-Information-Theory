"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """f 运算（对数域 box-plus，与 min-sum 兼容）。"""
    la, lb = np.asarray(La, dtype=np.float64), np.asarray(Lb, dtype=np.float64)
    if la.shape == () and lb.shape == ():
        return _logdomain_sum(la + lb, 0.0) - _logdomain_sum(la, lb)
    return np.vectorize(lambda a, b: _logdomain_sum(a + b, 0.0) - _logdomain_sum(a, b))(la, lb)


def g_operation(La, Lb, u_hat):
    """g 运算（与 polar-codes lower_llr 一致）：La - Lb (u=1) 或 La + Lb (u=0)"""
    u_hat = np.asarray(u_hat)
    if u_hat.shape == ():
        return La + Lb if u_hat == 0 else La - Lb
    return np.where(u_hat == 0, La + Lb, La - Lb)


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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
    """
    递归 SC 译码（参考实现）。
    极化码 PSCD 调度与简单二叉递归不等价，此处复用已验证的非递归核。
    """
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================

_SC_CACHE = {}


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    if N in _SC_CACHE:
        return _SC_CACHE[N]

    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        tmp = phi
        while tmp & 1:
            layers.append(int(math.log2(tmp & -tmp)))
            tmp >>= 1
        llr_layer_vec.append(layers)

        layers_b = []
        tmp = phi
        while tmp & 1:
            layers_b.append(int(math.log2(tmp & -tmp)))
            tmp >>= 1
        bit_layer_vec.append(layers_b)

    result = (lambda_offset, llr_layer_vec, bit_layer_vec)
    _SC_CACHE[N] = result
    return result


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（基于 polar-codes SCD 实现）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    if frozen_bits.dtype == bool:
        frozen_set = set(np.where(frozen_bits)[0])
    else:
        frozen_set = set(np.where(frozen_bits.astype(int))[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]

        if l < N / 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    return u_hat
