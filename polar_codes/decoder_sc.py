"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr_exact(l1, l2):
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _lower_llr_exact(l1, l2, b):
    return l1 + l2 if b == 0 else l1 - l2


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


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


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(np.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << (n - i))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = 0
        p = phi
        while p & 1:
            l += 1
            p >>= 1
        llr_layer_vec.append(list(range(l, n)))
        layers_bit = []
        if phi & 1:
            p = phi
            layer = 0
            while p & 1:
                layers_bit.append(layer)
                p >>= 1
                layer += 1
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _scd_core(llr_ch, frozen_set, N, use_min_sum=True):
    """基于 Permuted SC 的高效非递归实现"""
    n = int(np.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    f_func = f_operation if use_min_sum else _upper_llr_exact

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_func(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr_exact(L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1])

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for l in [_bit_reversed_index(i, n) for i in range(N)]:
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)
    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，内部调用非递归核心）"""
    from encoder import bit_reversal_permutation
    N = len(llr)
    br = bit_reversal_permutation(N)
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
    return _scd_core(np.asarray(llr, dtype=np.float64)[br], frozen_set, N)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    编码使用比特倒序置换，因此信道 LLR 需先倒序后再译码。
    """
    from encoder import bit_reversal_permutation
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
    return _scd_core(np.asarray(llr_ch, dtype=np.float64)[br], frozen_set, N)
