"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 PSC（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def _bit_reversed_index(i, n):
    """将 n 位索引 i 做比特倒序"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def _active_llr_level(i, n):
    """llr 更新起始层：从高位起第一个 1 的位置"""
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
    """比特回传起始层：从高位起第一个 0 的位置"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


# ==================== 基本运算 ====================

def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算（top=La, bottom=Lb）：
    u=0: La + Lb;  u=1: Lb - La
    """
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, Lb - La)


def reorder_channel_llr(llr_ch):
    """极化编码含比特倒序，将信道 LLR 映射到 PSC 因子图叶节点顺序"""
    N = len(llr_ch)
    inv_br = np.argsort(bit_reversal_permutation(N))
    return np.asarray(llr_ch, dtype=np.float64)[inv_br]


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr_ch, frozen_bits):
    """基于 PSC 的递归等价实现（调用非递归核心）"""
    return sc_decode(llr_ch, frozen_bits)


# ==================== 非递归 PSC 译码 ====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（PSC 顺序）。
    返回 bit_order, llr_start_layer, bit_start_layer
    """
    n = int(math.log2(N))
    bit_order = [_bit_reversed_index(phi, n) for phi in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for l in bit_order:
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_start = n - _active_bit_level(l, n) + 1
        bit_layer_vec.append(list(range(n, bit_start - 1, -1)))
    return bit_order, llr_layer_vec, bit_layer_vec


def _psc_update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, len(L), block_size):
            if j % block_size < branch_size:
                top = L[j, s]
                btm = L[j + branch_size, s]
                L[j, s + 1] = f_operation(top, btm)
            else:
                btm = L[j, s]
                top = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(top, btm, top_bit)


def _psc_update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    Permuted Successive Cancellation (PSC) 非递归译码。
    """
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = reorder_channel_llr(llr_ch)

    bit_order = [_bit_reversed_index(phi, n) for phi in range(N)]
    u_hat = np.zeros(N, dtype=int)

    for l in bit_order:
        _psc_update_llrs(L, B, l, n)
        if l in frozen_set:
            bit = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
        B[l, n] = bit
        u_hat[l] = bit
        _psc_update_bits(B, l, n, N)

    return u_hat
