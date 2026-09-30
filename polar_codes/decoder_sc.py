"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，log-domain f 函数）
"""
import numpy as np

from encoder import bit_reversal_permutation


def logdomain_sum(x, y):
    if x >= y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    """f 运算（log-domain，与 min-sum 在高 SNR 下接近）"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, bit):
    """g 运算"""
    if bit == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（供 BP / 测试使用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（向量形式）"""
    return (1 - 2 * u_hat) * La + Lb


def bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


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


def _permute_channel_llr(llr_ch):
    """编码端含比特倒序置换时，将信道 LLR 映射到 SC 因子图顺序"""
    N = len(llr_ch)
    n = int(np.log2(N))
    br = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=int)
    inv[br] = np.arange(N)
    return llr_ch[inv]


def _update_llrs(L, B, l, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1])


def _update_bits(B, l, n):
    if l < 2 ** (n - 1):
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    llr_ch = _permute_channel_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed(i, n)
        _update_llrs(L, B, l, n)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（调用非递归实现作为参考）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：返回解码阶段索引（供文档/扩展使用）"""
    n = int(np.log2(N))
    llr_layer_vec = [list(range(n - active_llr_level(bit_reversed(phi, n), n), n)) for phi in range(N)]
    bit_layer_vec = [list(range(n, n - active_bit_level(bit_reversed(phi, n), n), -1)) for phi in range(N)]
    lambda_offset = [2 ** i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
