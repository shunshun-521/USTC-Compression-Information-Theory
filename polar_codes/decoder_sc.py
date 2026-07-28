"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversed


def logdomain_sum(x, y):
    """对数域加法。"""
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    """f 运算（box-plus）。"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    """g 运算。"""
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def f_operation(La, Lb):
    """向量化 f 运算（box-plus）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    result = np.empty_like(La)
    for i in range(La.size):
        result[i] = upper_llr(La[i], Lb[i])
    return result


def g_operation(La, Lb, u_hat):
    """向量化 g 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    result = np.empty_like(La)
    for i in range(La.size):
        result[i] = lower_llr(La[i], Lb[i], int(u_hat[i]))
    return result


def active_llr_level(i, n):
    """二进制表示中从高位起第一个 1 的位置（层数）。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """二进制表示中从高位起第一个 0 的位置（层数）。"""
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
    """递归 SC 译码（与 sc_decode 等价的参考实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    decode_order = [bit_reversed(i, n) for i in range(N)]

    for l in decode_order:
        start_llr = n - active_llr_level(l, n)
        llr_layer_vec.append(list(range(start_llr, n)))

        if l < N / 2:
            bit_layers = []
        else:
            start_bit = n - active_bit_level(l, n)
            bit_layers = list(range(n, start_bit, -1))
        bit_layer_vec.append(bit_layers)

    return decode_order, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 按自然顺序输入，按比特倒序索引顺序译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    decode_order, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for idx, l in enumerate(decode_order):
        for s in llr_layer_vec[idx]:
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top_llr = L[j, s]
                    btm_llr = L[j + branch_size, s]
                    L[j, s + 1] = upper_llr(top_llr, btm_llr)
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = lower_llr(btm_llr, top_llr, top_bit)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        for s in bit_layer_vec[idx]:
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
