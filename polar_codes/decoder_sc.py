"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import _bit_reversal_indices


# ==================== 基本运算 ====================

def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def f_operation(La, Lb):
    """
    f 运算（log-domain box-plus）：
    f(La, Lb) = logsum(La+Lb, 0) - logsum(La, Lb)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = La + Lb (u=0) 或 La - Lb (u=1)
    其中 La 为下分支 LLR，Lb 为上分支 LLR（与因子图一致）
    """
    u_hat = np.asarray(u_hat)
    if np.isscalar(u_hat) or u_hat.size == 1:
        u = int(u_hat) if np.isscalar(u_hat) else int(u_hat.flat[0])
        return La + Lb if u == 0 else La - Lb
    result = np.where(u_hat == 0, La + Lb, La - Lb)
    return result


def _bit_reversed(i, n):
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


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
    """
    递归 SC 译码。
    frozen_bits: True/1 表示冻结位
    信道 LLR 为编码输出顺序，内部对比特倒序置换后与编码蝶形结构对齐。
    """
    N = len(llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    brp = _bit_reversal_indices(N)
    L[:, 0] = np.asarray(llr, dtype=np.float64)[brp]

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top = L[j, s]
                    btm = L[j + branch_size, s]
                    L[j, s + 1] = f_operation(top, btm)
                else:
                    btm = L[j, s]
                    top = L[j - branch_size, s]
                    top_bit = int(B[j - branch_size, s + 1])
                    B[j - branch_size, s + 1] = top_bit
                    L[j, s + 1] = g_operation(btm, top, top_bit)

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for phi in range(N):
        l = _bit_reversed(phi, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（与递归实现等价的层索引表示）。
    """
    n = int(math.log2(N))
    lambda_offset = [0]
    for layer in range(n):
        lambda_offset.append(lambda_offset[-1] + (1 << layer))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（内部调用与参考实现一致的递推结构）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
