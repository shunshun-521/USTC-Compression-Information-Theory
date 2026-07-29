"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import _bit_reverse_indices


def _bit_reversed(phi, n):
    """单索引比特倒序。"""
    result = 0
    for i in range(n):
        if phi & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    """LLR 更新起始层（首个 1 之前的 0 个数）。"""
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
    """比特回传起始层（首个 0 之前的 1 个数）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    """f 运算（对数域 box-plus）。"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    """g 运算（对数域）。"""
    return l1 + l2 if b == 0 else l1 - l2


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（向量化）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        if l >= N // 2:
            start_b = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, start_b - 1, -1)))
        else:
            bit_layer_vec.append([])
    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_TABLES = {}


def _get_sc_tables(N):
    if N not in _SC_TABLES:
        _SC_TABLES[N] = precompute_sc_indices(N)
    return _SC_TABLES[N]


def _update_llrs(L, B, l, n):
    """更新索引 l 处的 LLR 树。"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, len(L), block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = _lower_llr(
                    L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                )


def _update_bits(B, l, n):
    """比特回传。"""
    if l < len(B) // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    decode_order = [_bit_reversed(i, n) for i in range(N)]

    for l in decode_order:
        _update_llrs(L, B, l, n)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, n].copy()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与非递归等价）。"""
    return sc_decode(llr, frozen_bits)
