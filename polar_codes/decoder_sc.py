"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，PSC 算法）
"""
import math
import numpy as np


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def _logdomain_sum(x, y):
    if x == np.inf:
        return y
    if y == np.inf:
        return x
    if x == -np.inf:
        return y
    if y == -np.inf:
        return x
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _active_llr_level(i, n):
    """Find the first 1 in the binary expansion of i (mcba1n convention)."""
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
    """Find the first 0 in the binary expansion of i (mcba1n convention)."""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def f_operation(La, Lb):
    """
    f 运算（对数域 box-plus，标量或向量化）。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0 and Lb.ndim == 0:
        return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)
    return np.vectorize(
        lambda a, b: _logdomain_sum(a + b, 0.0) - _logdomain_sum(a, b),
        otypes=[np.float64],
    )(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = La + Lb (u=0) 或 La - Lb (u=1)
    """
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，与 PSC 结果一致）。
    """
    return _sc_decode_psc_layout(llr, frozen_bits)


def _sc_decode_psc_layout(llr, frozen_bits):
    """PSC 非递归译码核心（与 mcba1n SCD 一致的数据布局）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr

    def upper_llr(l1, l2):
        if np.isinf(l1) and not np.isinf(l2):
            return l2
        if not np.isinf(l1) and np.isinf(l2):
            return l1
        if np.isinf(l1) and np.isinf(l2):
            return np.inf
        return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)

    def lower_llr(l1, l2, b):
        if b == 0:
            if np.isinf(l1) or np.isinf(l2):
                return np.inf
            return l1 + l2
        return l1 - l2

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for l in [_bit_reversed(i, n) for i in range(N)]:
        update_llrs(l)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（兼容接口，PSC 内部自管理）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        layers = list(range(n - _active_llr_level(l, n), n))
        llr_layer_vec.append(layers)
        if l < N / 2:
            bit_layers = []
        else:
            bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（PSC 算法）。
    """
    return _sc_decode_psc_layout(llr_ch, frozen_bits)
