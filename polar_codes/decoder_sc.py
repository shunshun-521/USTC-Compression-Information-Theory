"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（向量化）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def _log_sum(x, y):
    m = np.maximum(x, y)
    return m + np.log1p(np.exp(-np.abs(x - y)))


def f_operation_exact(La, Lb):
    """精确 box-plus（log-domain）"""
    return _log_sum(La + Lb, 0.0) - _log_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


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


def _upper_llr(l1, l2):
    return f_operation_exact(l1, l2)


def _lower_llr(btm_llr, top_llr, b):
    if b == 0:
        return btm_llr + top_llr
    return btm_llr - top_llr


def _update_llrs(l, n, L, B, N):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = _lower_llr(
                    L[j, s],
                    L[j - branch_size, s],
                    int(B[j - branch_size, s + 1]),
                )


def _update_bits(l, n, B, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _sc_decode_core(llr_ch, frozen_set, n, N):
    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = llr_ch
    br = bit_reversal_permutation(N)

    for i in range(N):
        l = br[i]
        _update_llrs(l, n, L, B, N)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(l, n, B, N)

    return B[:, n].astype(np.int8)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（自然序信道 LLR）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    br = bit_reversal_permutation(N)
    # 与含比特倒序的编码器配套：先对信道 LLR 倒序
    llr_work = llr_ch[br]
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int) > 0)[0])
    return _sc_decode_core(llr_work, frozen_set, n, N)


def sc_decode_recursive(llr, frozen_bits):
    """递归版本接口（调用非递归核心）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(np.log2(N))
    br = bit_reversal_permutation(N)
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = br[i]
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
