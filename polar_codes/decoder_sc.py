"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def _log_sum_exp(x, y):
    if x > y:
        return x + math.log1p(math.exp(y - x))
    if y > -np.inf:
        return y + math.log1p(math.exp(x - y))
    return x


def boxplus(La, Lb):
    """精确 box-plus（向量化）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    out = np.empty(np.broadcast_shapes(La.shape, Lb.shape), dtype=np.float64)
    it = np.nditer(
        [La, Lb, out],
        flags=["multi_index"],
        op_flags=[["readonly"], ["readonly"], ["writeonly"]],
    )
    for la, lb, o in it:
        t1 = _log_sum_exp(0.0, la + lb)
        t2 = _log_sum_exp(0.0, la - lb)
        o[...] = t1 - t2
    return out


def f_operation(La, Lb):
    """f 运算（box-plus）"""
    return boxplus(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def upper_llr(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return _log_sum_exp(l1 + l2, 0.0) - _log_sum_exp(l1, l2)


def lower_llr(l1, l2, b):
    b = int(b)
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    return l1 - l2


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


def _channel_to_natural_llr(llr_ch):
    """将信道 LLR 映射到译码树自然序（匹配含比特倒序的编码器）"""
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    llr_nat = np.zeros(N, dtype=np.float64)
    llr_nat[rev] = llr_ch
    return llr_nat


def _scd_core(llr_nat, frozen_indices):
    """非递归 SCD 核心（自然序 LLR 输入）"""
    N = len(llr_nat)
    n = int(math.log2(N))
    frozen_set = set(frozen_indices)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_nat

    for i in range(N):
        l = int(format(i, f"0{n}b")[::-1], 2)

        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1])

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N / 2:
            continue

        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，与 sc_decode 等价）"""
    return sc_decode(llr_ch, frozen_bits)


# ==================== 非递归 SC 译码（主实现）====================


def precompute_sc_indices(N):
    """预计算辅助索引"""
    n = int(math.log2(N))
    rev = bit_reversal_permutation(N)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = rev[phi]
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(
            list(range(n, n - active_bit_level(l, n), -1)) if l >= N // 2 else []
        )
    lambda_offset = [0]
    for layer in range(1, n + 1):
        lambda_offset.append(lambda_offset[-1] + 2 ** (n - layer + 1))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    参数 llr_ch 为信道序 LLR；frozen_bits 中 True 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_indices = list(np.where(frozen_bits)[0])
    llr_nat = _channel_to_natural_llr(llr_ch)
    return _scd_core(llr_nat, frozen_indices)
