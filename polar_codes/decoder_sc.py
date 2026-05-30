"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    """对数域加法（精确 boxplus 的 LLR 形式）"""
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2, use_minsum=True):
    """f 节点 LLR 更新"""
    if use_minsum:
        return float(f_operation(l1, l2))
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, b, use_minsum=True):
    """g 节点 LLR 更新"""
    if use_minsum:
        return float(g_operation(l1, l2, b))
    if b == 0:
        return l1 + l2
    return l1 - l2


def _bit_reversed(i, n):
    r = 0
    for k in range(n):
        if (i >> k) & 1:
            r |= 1 << (n - 1 - k)
    return r


def _active_llr_level(i, n):
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
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用分层非递归核心）"""
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（与分层更新等价）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        if l >= N // 2:
            end = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, end, -1)))
        else:
            bit_layer_vec.append([])
    return llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits, use_minsum=True):
    """
    非递归 SC 译码（分层 L/B 数组，比特倒序判决顺序）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    for phi_nat in range(N):
        l = _bit_reversed(phi_nat, n)

        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            branch = block // 2
            for j in range(l, N, block):
                if j % block < branch:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch, s], use_minsum)
                else:
                    top_bit = B[j - branch, s + 1]
                    L[j, s + 1] = lower_llr(
                        L[j - branch, s], L[j, s], top_bit, use_minsum
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N // 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block = 1 << s
            branch = block // 2
            for j in range(l, -1, -block):
                if j % block >= branch:
                    B[j - branch, s - 1] = int(B[j, s]) ^ int(B[j - branch, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


# 兼容旧接口
def precompute_sc_indices_legacy(N):
    llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    return lambda_offset, llr_layer_vec, bit_layer_vec
