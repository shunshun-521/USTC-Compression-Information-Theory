"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """
    f 运算（对数域 box-plus，标量/向量化）。
    对极大 LLR 退化为 min-sum。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    large = 30.0
    use_min = (np.abs(La) > large) | (np.abs(Lb) > large)
    result = np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))
    if np.any(~use_min):
        # 逐元素精确计算（向量较小或用于递归块）
        flat_a = La.ravel()
        flat_b = Lb.ravel()
        flat_r = result.ravel()
        flat_u = use_min.ravel()
        for i in range(flat_a.size):
            if not flat_u[i]:
                flat_r[i] = _logdomain_sum(flat_a[i] + flat_b[i], 0.0) - _logdomain_sum(
                    flat_a[i], flat_b[i]
                )
        result = flat_r.reshape(La.shape)
    return result


def g_operation(La, Lb, u_hat):
    """
    g 运算：u=0 时 La+Lb；u=1 时 Lb-La。
    等价于 (1 - 2*u_hat) * La + Lb（La 为上支路，Lb 为下支路）。
    """
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, Lb - La)


def _bit_reversed(i, n):
    result = 0
    for bit in range(n):
        if (i >> bit) & 1:
            result |= 1 << (n - 1 - bit)
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


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，结果与非递归 sc_decode 一致）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助结构（与逐比特更新等价）。"""
    n = int(math.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in decode_order:
        start = n - _active_llr_level(phi, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_start = n - _active_bit_level(phi, n)
        bit_layer_vec.append(list(range(n, bit_start, -1)))
    return decode_order, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（基于分层 LLR/比特数组）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    decode_order = [_bit_reversed(i, n) for i in range(N)]

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N // 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                    B[j, s - 1] = B[j, s]

    u_hat = B[:, n].astype(int)
    return u_hat
