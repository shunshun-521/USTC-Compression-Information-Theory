"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================

def _sign_llr(x):
    """LLR 符号：非负为 +1，负为 -1"""
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return _sign_llr(La) * _sign_llr(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr_exact(l1, l2):
    """精确 log-domain f 运算（box-plus）"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr_exact(l1, l2, b):
    """精确 log-domain g 运算"""
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


def _init_llr_buffer(llr_ch):
    """将信道 LLR 填入 SC 树 stage 0（与比特倒序编码对齐）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    for i in range(N):
        L[br[i], 0] = llr_ch[i]
    return L, n


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr_natural, frozen_bits):
    """
    递归 SC 译码（自然序 LLR）。
    frozen_bits: 1 表示冻结位
    """
    llr_natural = np.asarray(llr_natural, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_natural)

    if N == 1:
        if frozen_bits[0]:
            return np.array([0], dtype=int)
        return np.array([0 if llr_natural[0] >= 0 else 1], dtype=int)

    half = N // 2
    llr_left = f_operation(llr_natural[:half], llr_natural[half:])
    u_left = sc_decode_recursive(llr_left, frozen_bits[:half])
    llr_right = g_operation(llr_natural[:half], llr_natural[half:], u_left)
    u_right = sc_decode_recursive(llr_right, frozen_bits[half:])
    return np.concatenate([u_left, u_right])


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))

        if l < N // 2:
            bit_layers = []
        else:
            start_bit = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, start_bit, -1)))
            continue
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_natural, frozen_bits):
    """基于 mcba1n SCD 的非递归实现（自然序 LLR）"""
    llr_natural = np.asarray(llr_natural, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_natural)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_natural

    frozen_set = set(np.where(frozen_bits == 1)[0])

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top_llr = L[j, s]
                    btm_llr = L[j + branch_size, s]
                    L[j, s + 1] = _upper_llr_exact(top_llr, btm_llr)
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = _lower_llr_exact(btm_llr, top_llr, top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。

    参数：
        llr_ch: 长度 N 的信道接收 LLR
        frozen_bits: 长度 N，1 表示冻结位

    返回：
        u_hat: 长度 N 的估计源序列
    """
    L, n = _init_llr_buffer(llr_ch)
    llr_natural = L[:, 0].copy()
    # 自然序 LLR 用于递归参考译码路径
    return sc_decode_nonrecursive(llr_natural, frozen_bits)


def sc_decode_recursive_wrapper(llr_ch, frozen_bits):
    """递归 SC 译码（含 LLR 重排）"""
    L, _ = _init_llr_buffer(llr_ch)
    return sc_decode_recursive(L[:, 0], frozen_bits)
