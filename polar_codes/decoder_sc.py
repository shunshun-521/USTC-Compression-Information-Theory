"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted SCD）
"""
import numpy as np
from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """f 运算（对数域精确形式，SC 主路径）"""
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def f_operation_min_sum(La, Lb):
    """min-sum 近似的 f 运算（供 BP 等模块复用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（u_hat=0 -> La+Lb, u_hat=1 -> La-Lb）"""
    u = int(u_hat)
    return La + Lb if u == 0 else La - Lb


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
    return int(bit_reversal_permutation(1 << n)[i]) if (1 << n) == len(bit_reversal_permutation(1 << n)) else _br_scalar(i, n)


def _br_scalar(x, n):
    result = 0
    for k in range(n):
        if x & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


# ==================== 非递归 SC 译码（主实现）====================


def precompute_sc_indices(N):
    """预计算译码相位顺序（比特倒序）及层数辅助信息"""
    n = int(np.log2(N))
    phase_order = [_br_scalar(i, n) for i in range(N)]
    llr_layer_vec = [_active_llr_level(l, n) for l in phase_order]
    bit_layer_vec = [_active_bit_level(l, n) for l in phase_order]
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = int(B[j - branch_size, s + 1])
                L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], top_bit)


def _update_bits(B, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SC 译码。
    frozen_bits: True/1 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for i in range(N):
        l = _br_scalar(i, n)
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(np.int8)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（调用主实现，接口兼容）"""
    return sc_decode(llr, frozen_bits)
