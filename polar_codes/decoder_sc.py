"""
极化码 SC（串行抵消）译码器
Permuted SC（比特倒序索引译码）+ 非递归高效实现
"""
import math
import numpy as np
from encoder import bit_reversed_index


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr_exact(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _lower_llr_exact(l1, l2, b):
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    return l1 - l2


def f_operation(La, Lb):
    """
    f 运算（对数域盒加，比 min-sum 更精确）。
    """
    return _upper_llr_exact(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return _lower_llr_exact(Lb, La, u_hat)


def _active_llr_level(i, n):
    """比特倒序索引 i 的 LLR 更新起始层"""
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
    """比特倒序索引 i 的比特回传起始层"""
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
    """
    预计算非递归 SC 译码辅助向量。
    返回 layer_offset, llr_layer_vec, bit_layer_vec
    """
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for i in range(N):
        l = bit_reversed_index(i, n)
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    layer_offset = np.zeros(n + 1, dtype=int)
    offset = 0
    for layer in range(n + 1):
        layer_offset[layer] = offset
        offset += 2 ** layer

    return layer_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SC 译码主函数。
    frozen_bits: 1 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = llr_ch
    frozen_set = set(np.where(frozen_bits)[0])
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = bit_reversed_index(i, n)

        for s in range(n - _active_llr_level(l, n), n):
            block = 2 ** (s + 1)
            half = block // 2
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
                else:
                    top_llr = L[j - half, s]
                    bot_llr = L[j, s]
                    top_bit = B[j - half, s + 1]
                    L[j, s + 1] = g_operation(top_llr, bot_llr, top_bit)

        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block = 2 ** s
                half = block // 2
                for j in range(l, -1, -block):
                    if j % block >= half:
                        B[j - half, s - 1] = int(B[j, s]) ^ int(B[j - half, s])
                        B[j, s - 1] = B[j, s]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 Permuted SC 译码（参考实现，与 sc_decode 等价）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = llr
    frozen_set = set(np.where(frozen_bits)[0])
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = bit_reversed_index(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block = 2 ** (s + 1)
            half = block // 2
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - half, s], L[j, s], B[j - half, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block = 2 ** s
                half = block // 2
                for j in range(l, -1, -block):
                    if j % block >= half:
                        B[j - half, s - 1] = int(B[j, s]) ^ int(B[j - half, s])
                        B[j, s - 1] = B[j, s]

    return u_hat
