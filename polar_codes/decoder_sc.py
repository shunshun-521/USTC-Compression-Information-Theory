"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def bit_reversed(i, n):
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def active_llr_level(i, n):
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
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(btm_llr, top_llr, u_hat):
    """g 运算（lower_llr）：btm + top (u=0) 或 btm - top (u=1)"""
    if np.isscalar(u_hat) or np.ndim(u_hat) == 0:
        return btm_llr + top_llr if u_hat == 0 else btm_llr - top_llr
    out = np.empty_like(btm_llr, dtype=np.float64)
    mask0 = u_hat == 0
    out[mask0] = btm_llr[mask0] + top_llr[mask0]
    out[~mask0] = btm_llr[~mask0] - top_llr[~mask0]
    return out


def _prepare_llr(llr_ch):
    """编码端做了比特倒序，信道 LLR 需倒序后送入译码树"""
    N = len(llr_ch)
    n = int(np.log2(N))
    rev = np.array([bit_reversed(i, n) for i in range(N)], dtype=int)
    return llr_ch.astype(np.float64)[rev]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    N = len(llr)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = _prepare_llr(llr)

    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = bit_reversed(i, n)
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s],
                        int(B[j - branch_size, s + 1]),
                    )

        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（与 SCD 算法对应）"""
    n = int(np.log2(N))
    decode_order = [bit_reversed(i, n) for i in range(N)]

    llr_layer_vec = []
    bit_layer_vec = []

    for l in decode_order:
        start = n - active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))

        if l < N // 2:
            bit_layer_vec.append([])
        else:
            bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))

    return decode_order, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    decode_order, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = _prepare_llr(llr_ch)

    u_hat = np.zeros(N, dtype=int)

    for idx, l in enumerate(decode_order):
        for s in llr_layer_vec[idx]:
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s],
                        int(B[j - branch_size, s + 1]),
                    )

        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit

        for s in bit_layer_vec[idx]:
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return u_hat
