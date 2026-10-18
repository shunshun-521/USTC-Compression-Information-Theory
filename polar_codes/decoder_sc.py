"""
极化码 SC（串行抵消）译码器
非递归 SCD（与 polarcodes 一致的 LLR/比特表更新）+ 递归参考实现
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u_hat 可为标量或数组"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def active_llr_level(i, n):
    """polarcodes: 从 MSB 起第一个 1 之前 0 的个数 + 1"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """polarcodes: 从 MSB 起第一个 0 之前 1 的个数 + 1"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（自然比特顺序输出）。
    frozen_bits[i]=True/1 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for l in [bit_reversed(i, n) for i in range(N)]:
        _update_llrs(L, B, l, n)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]
        _update_bits(B, l, n)

    return u_hat


def _update_llrs(L, B, l, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                top_llr = L[j, s]
                btm_llr = L[j + branch_size, s]
                L[j, s + 1] = f_operation(top_llr, btm_llr)
            else:
                btm_llr = L[j, s]
                top_llr = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)


def _update_bits(B, l, n):
    if l < B.shape[0] // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 各比特相位所需的 LLR/比特更新层列表（SCD 顺序）。
    """
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        if l < N // 2:
            bit_layer_vec.append([])
        else:
            bit_layer_vec.append(
                list(range(n, n - active_bit_level(l, n), -1))
            )
    lambda_offset = [0] * N
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与分层 f/g 配对一致）"""
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def sc(llrs, indices):
        half = len(indices) // 2
        if half == 0:
            i = indices[0]
            u_hat[i] = 0 if frozen_bits[i] or llrs[0] >= 0 else 1
            return
        left_llr = [f_operation(llrs[i], llrs[i + half]) for i in range(half)]
        sc(left_llr, indices[:half])
        enc_left = [u_hat[i] for i in indices[:half]]
        right_llr = [g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)]
        sc(right_llr, indices[half:])

    sc(llr.tolist(), list(range(N)))
    return u_hat
