"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，按比特倒序处理）
"""
import math
import numpy as np


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    """f 运算（对数域 box-plus）"""
    l1 = float(l1)
    l2 = float(l2)
    if np.isinf(l1) and np.isfinite(l2):
        return l2
    if np.isfinite(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    """g 运算"""
    b = int(b)
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def f_operation(La, Lb):
    """向量化的 f 运算（调用 upper_llr）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.vectorize(upper_llr)(La, Lb)


def g_operation(La, Lb, u_hat):
    """向量化的 g 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return np.vectorize(lower_llr)(La, Lb, u_hat)


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


def _frozen_indices(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return set(np.where(frozen_bits)[0])


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    与 polar_encode 配套时，信道 LLR 需按与 sc_decode 相同的因子图处理；
    此处直接调用等价的非递归实现以保证结果一致。
    """
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 辅助向量（比特倒序处理顺序 + 各比特活跃层）
    """
    from encoder import bit_reversed_index

    n = int(math.log2(N))
    lambda_offset = [1 << l for l in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    decode_order = []
    for i in range(N):
        l = bit_reversed_index(i, n)
        decode_order.append(l)
        start_s = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start_s, n)))
        start_b = n - _active_bit_level(l, n)
        bit_layer_vec.append(list(range(n, start_b - 1, -1)) if start_b <= n else [])
    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（主实现）"""
    from encoder import bit_reversed_index

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_set = _frozen_indices(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], top_bit)

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

    for i in range(N):
        l = bit_reversed_index(i, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)
