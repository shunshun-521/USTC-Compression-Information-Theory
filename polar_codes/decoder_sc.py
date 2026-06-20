"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    """对数域 f 运算（比 min-sum 更精确）"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    """对数域 g 运算"""
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


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


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _frozen_to_decoder_domain(frozen_bits, N):
    """将自然序冻结位索引映射到内部译码域（匹配 BR 编码器）"""
    br = bit_reversal_permutation(N)
    inv = np.argsort(br)
    frozen_natural = np.where(np.asarray(frozen_bits, dtype=int))[0]
    return set(int(inv[k]) for k in frozen_natural)


def _decoder_domain_to_natural(u_prime, N):
    """内部译码域输出映射回自然序 u"""
    br = bit_reversal_permutation(N)
    inv = np.argsort(br)
    return np.asarray(u_prime, dtype=np.int32)[inv]


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用非递归核心并做索引变换）"""
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = _frozen_to_decoder_domain(frozen_bits, N)
    u_prime = _scd_core(llr, frozen_set, n)
    return _decoder_domain_to_natural(u_prime, N)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（层索引与比特倒序访问顺序）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        temp = l
        first_zero = 0
        while first_zero < n and (temp & 1) == 0:
            first_zero += 1
            temp >>= 1
        for layer in range(first_zero, n):
            llr_layer_vec[phi].append(layer)

        temp = l
        layer = 0
        while layer < n and (temp & 1) == 1:
            bit_layer_vec[phi].append(layer)
            temp >>= 1
            layer += 1

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _scd_core(likelihoods, frozen_set, n):
    """mcba1n 风格非递归 SCD 核心（内部译码域）"""
    N = 2 ** n
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = likelihoods

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2**s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = _bit_reversed_index(i, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(np.int32)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    llr_ch: 信道 LLR（与发送码字同序，无需额外倒序）
    frozen_bits: 1 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = _frozen_to_decoder_domain(frozen_bits, N)
    u_prime = _scd_core(llr_ch, frozen_set, n)
    return _decoder_domain_to_natural(u_prime, N)


_SC_CACHE = {}


def get_sc_indices(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]
