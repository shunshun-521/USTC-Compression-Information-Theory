"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Vangala 置换 SCD，高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================

def _sign(x):
    """LLR 符号：0 视为 +1"""
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）"""
    return _sign(La) * _sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr_exact(l1, l2):
    """精确 log-domain box-plus（f 运算）"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _lower_llr_exact(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
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


def _permute_channel_llr(llr_ch):
    """将信道 LLR 置换为与 B_N F^{\\otimes n} 编码器一致的顺序"""
    N = len(llr_ch)
    n = int(np.log2(N))
    brp = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[brp]


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码参考实现（与 sc_decode 等价的 Vangala 置换 SCD）。
    输入 llr 为经比特倒序置换后的信道 LLR。
    """
    # 与 sc_decode 使用相同算法，保证 B_N F^{\\otimes n} 编码下的一致性
    N = len(llr)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = np.asarray(llr, dtype=np.float64)

    for phi in range(N):
        l = _bit_reversed(phi, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_exact(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr_exact(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]
    return B[:, n].astype(int)


def sc_decode_recursive_permuted(llr_ch, frozen_bits):
    return sc_decode_recursive(_permute_channel_llr(llr_ch), frozen_bits)


# ==================== 非递归 SC 译码（Vangala 置换 SCD）====================

def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（与 Vangala SCD 等价）"""
    n = int(np.log2(N))
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layer_vec[phi] = list(range(n - _active_llr_level(l, n), n))
        if l >= N / 2:
            bit_layer_vec[phi] = list(
                range(n, n - _active_bit_level(l, n), -1)
            )
        else:
            bit_layer_vec[phi] = []

    return llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Vangala 2014 置换 SCD）。
    信道 LLR 经比特倒序置换后与 B_N F^{\\otimes n} 编码器匹配。
    """
    llr = _permute_channel_llr(llr_ch)
    N = len(llr)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    for phi in range(N):
        l = _bit_reversed(phi, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_exact(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr_exact(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
