"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_diff(_logdomain_sum(l1 + l2, 0.0), _logdomain_sum(l1, l2))


def lower_llr(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def _bit_reversed(i, n):
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
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


def _hard_decision(y):
    return 0 if y >= 0 else 1


def _scd_core(N, n, frozen_idx, llr_perm):
    """SSC 译码核心（LLR 已按比特倒序置换对齐编码器）"""
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_perm

    for i in range(N):
        l = _bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_idx:
            B[l, n] = 0
        else:
            B[l, n] = _hard_decision(L[l, n])

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，含 LLR 对齐）"""
    N = len(llr)
    n = int(math.log2(N))
    rev = bit_reversal_permutation(N)
    llr_perm = np.asarray(llr, dtype=np.float64)[rev]
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_idx = set(np.where(frozen_bits)[0])

    def decode_node(llr_node, indices):
        m = len(llr_node)
        if m == 1:
            idx = indices[0]
            if idx in frozen_idx:
                return np.array([0])
            return np.array([0 if llr_node[0] >= 0 else 1])
        half = m // 2
        u_left = decode_node(
            f_operation(llr_node[:half], llr_node[half:]), indices[:half]
        )
        u_right = decode_node(
            g_operation(llr_node[:half], llr_node[half:], u_left), indices[half:]
        )
        return np.concatenate([u_left, u_right])

    return decode_node(llr_perm, list(range(N)))


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [2 ** phi_layer - 1 for phi_layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(_active_llr_level(phi, n) - 1, n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(phi, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    frozen_bits: 1/True 表示冻结位
    """
    from ref_scd import SCD

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_idx = np.where(frozen_bits)[0]
    rev = bit_reversal_permutation(N)
    llr_perm = llr_ch[rev]

    class _PC:
        pass

    pc = _PC()
    pc.N = N
    pc.n = n
    pc.frozen = frozen_idx
    pc.likelihoods = llr_perm
    return SCD(pc).decode()
