"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）与非递归 Permuted SCD（主实现，与 BR 编码配套）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """f 运算（对数域 boxplus）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0 and Lb.ndim == 0:
        return logdomain_sum(La + Lb, 0.0) - logdomain_sum(La, Lb)
    return np.vectorize(logdomain_sum)(La + Lb, 0.0) - np.vectorize(logdomain_sum)(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：b=0 -> l1+l2, b=1 -> l1-l2（La 为上支路，Lb 为下支路）"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if (x >> i) & 1:
            result |= 1 << (n - 1 - i)
    return result


def _update_llr(L, B, phi, n, N):
    for j in range(n - 1, -1, -1):
        s = 2 ** (n - j)
        t = s // 2
        for i in range(phi, N, s):
            if t > i % s:
                L[i, j] = f_operation(L[i, j + 1], L[i + t, j + 1])
            else:
                L[i, j] = g_operation(L[i, j + 1], L[i - t, j + 1], B[i - t, j])


def _update_bits(B, phi, n, N):
    active = [phi]
    for j in range(n):
        s = 2 ** (n - j)
        t = s // 2
        nxt = []
        for i in active:
            if t <= i % s:
                B[i - t, j + 1] = (B[i, j] ^ B[i - t, j]) & 1
                B[i, j + 1] = B[i, j]
                nxt.extend([i, i - t])
        active = nxt


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    rev = bit_reversal_permutation(len(llr))
    return sc_decode(llr[rev], frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助结构（文档接口）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        bit_layers = []
        mask = 2 ** (n - 1)
        while mask:
            if (l & mask) == 0:
                bit_layers.append(int(math.log2(mask)))
            mask >>= 1
        bit_layer_vec.append(bit_layers)
        llr_layer_vec.append(list(range(n)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SCD。
    输入信道 LLR 为编码输出 x 的顺序；内部对比特倒序索引译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    frozen_set = set(np.where(frozen_bits)[0])

    rev = bit_reversal_permutation(N)
    llr = llr_ch[rev]

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, n] = llr
    u_hat = np.zeros(N, dtype=np.int8)

    for i in range(N):
        l = _bit_reversed_index(i, n)
        _update_llr(L, B, l, n, N)
        if l in frozen_set:
            u_hat[l] = 0
        else:
            u_hat[l] = 0 if L[l, 0] >= 0 else 1
        B[l, 0] = u_hat[l]
        _update_bits(B, l, n, N)

    return u_hat.astype(int)
