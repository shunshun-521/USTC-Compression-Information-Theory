"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 Permuted SCD（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================

def logdomain_sum(a, b):
    """对数域加法。"""
    if a >= b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


def f_boxplus(La, Lb):
    """精确 box-plus f 运算（对数域）。"""
    La = np.asarray(La, dtype=float)
    Lb = np.asarray(Lb, dtype=float)
    return logdomain_sum(La + Lb, 0.0) - logdomain_sum(La, Lb)


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（默认 SC 使用精确 box-plus，此函数保留供 BP 等模块使用）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=float)
    Lb = np.asarray(Lb, dtype=float)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    La = np.asarray(La, dtype=float)
    Lb = np.asarray(Lb, dtype=float)
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def reorder_channel_llr(llr_ch):
    """将信道 LLR 按比特倒序重排，供 SC 树使用。"""
    llr_ch = np.asarray(llr_ch, dtype=float)
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br]


def active_llr_level(i, n):
    """求 i 的二进制表示中首个 1 的位置（从高位计）。"""
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
    """求 i 的二进制表示中首个 0 的位置（从高位计）。"""
    mask = 2 ** (n - 1)
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
    递归 SC 译码（使用精确 box-plus）。
    frozen_bits: 1 表示冻结位
    """
    llr = np.asarray(llr, dtype=float)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_boxplus(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


# ==================== 非递归 SC 译码（Permuted SCD）====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回 decode_order, n
    """
    n = int(np.log2(N))
    br = bit_reversal_permutation(N)
    decode_order = [br[i] for i in range(N)]
    return decode_order, n


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SC 译码主函数。

    参数：
        llr_ch: 长度 N 的信道接收 LLR（自然顺序，对应编码后的码字位）
        frozen_bits: 长度 N 数组，1 表示冻结位，0 表示信息位

    返回：
        u_hat: 长度 N 的估计源序列
    """
    llr_ch = np.asarray(llr_ch, dtype=float)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_set = set(np.where(frozen_bits == 1)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    # 编码含比特倒序置换，信道 LLR 需重排后再送入 SC 树
    L[:, 0] = reorder_channel_llr(llr_ch)

    u_hat = np.zeros(N, dtype=int)
    decode_order, _ = precompute_sc_indices(N)

    for l in decode_order:
        _update_llrs(L, B, l, n, f_boxplus, g_operation)

        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit

        _update_bits(B, l, n)

    return u_hat


def _update_llrs(L, B, l, n, f_fn, g_fn):
    """更新 LLR 树（Permuted SCD）。"""
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                top = L[j, s]
                btm = L[j + branch_size, s]
                L[j, s + 1] = f_fn(top, btm)
            else:
                btm = L[j, s]
                top = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                if np.isnan(top_bit):
                    top_bit = 0
                L[j, s + 1] = g_fn(top, btm, top_bit)


def _update_bits(B, l, n):
    """比特回传（Permuted SCD）。"""
    if l < B.shape[0] / 2:
        return

    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                bj = 0 if np.isnan(B[j, s]) else int(B[j, s])
                btop = 0 if np.isnan(B[j - branch_size, s]) else int(B[j - branch_size, s])
                B[j - branch_size, s - 1] = bj ^ btop
                B[j, s - 1] = bj
