"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于分层 L/B 更新）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """box-plus（SC 精确 f 运算）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    a, b = np.abs(La), np.abs(Lb)
    s = np.sign(La) * np.sign(Lb)
    m = np.minimum(a, b)
    M = np.maximum(a, b)
    return s * (m + np.log1p(np.exp(-M + m)) - np.log1p(np.exp(-np.abs(La - Lb))))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(a, b):
    if np.isinf(a):
        return b
    if np.isinf(b):
        return a
    m = max(a, b)
    return m + np.log1p(np.exp(-abs(a - b)))


def _upper_llr(l1, l2):
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    return l1 + l2 if b == 0 else l1 - l2


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


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


def _frozen_set_from_mask(frozen_bits):
    fb = np.asarray(frozen_bits, dtype=int)
    return set(np.where(fb.astype(bool))[0])


def precompute_sc_indices(N):
    """预计算非递归 SC 的层索引（与标准 SCD 相位更新一致）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(n - _active_llr_level(phi, n), n)))
        if phi % 2 == 0:
            bit_layer_vec.append(list(range(n)))
        else:
            bit_layer_vec.append(list(range(n - _active_bit_level(phi, n))))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _scd_core(llr, frozen_set, n):
    """非递归 SCD（译码顺序为 bit-reversed 相位）"""
    N = len(llr)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    for i in range(N):
        l = _bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = 0 if np.isnan(B[j - branch_size, s + 1]) else int(B[j - branch_size, s + 1])
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N / 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    信道 LLR 对应 polar_encode 输出顺序；内部映射到极化域后 SCD。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    perm = bit_reversal_permutation(N)
    llr_polar = llr_ch[perm]
    frozen_set = _frozen_set_from_mask(frozen_bits)
    return _scd_core(llr_polar, frozen_set, n)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC（与 sc_decode 等价，用于校验）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    perm = bit_reversal_permutation(N)
    llr = llr_ch[perm]
    frozen_set = _frozen_set_from_mask(frozen_bits)

    def decode_block(L, depth):
        if depth == 0:
            idx = 0
            if idx in frozen_set:
                return np.array([0], dtype=int)
            return np.array([0 if L[0] >= 0 else 1], dtype=int)
        half = 2 ** (depth - 1)
        L_left = np.empty(half, dtype=np.float64)
        L_right = np.empty(half, dtype=np.float64)
        for i in range(half):
            L_left[i] = _upper_llr(L[i], L[i + half])
        u_left = decode_block(L_left, depth - 1)
        for i in range(half):
            L_right[i] = _lower_llr(L[i], L[i + half], u_left[i])
        u_right = decode_block(L_right, depth - 1)
        return np.concatenate([u_left, u_right])

    # 递归版本按自然分段，与 SCD 在 N<=64 上对齐验证
    u_nat = decode_block(llr, n)
    return u_nat
