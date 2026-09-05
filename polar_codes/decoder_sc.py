"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于 Permuted SC）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def bit_reversed_index(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def active_llr_level(i, n):
    """LLR 更新起始层：二进制表示中首个 1 的位置（从高位计）"""
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
    """比特回传起始层：二进制表示中首个 0 的位置（从高位计）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u=0 -> La+Lb, u=1 -> La-Lb（支持向量化）"""
    u_hat = np.asarray(u_hat)
    if u_hat.ndim == 0:
        return La + Lb if u_hat == 0 else La - Lb
    return np.where(u_hat == 0, La + Lb, La - Lb)


def upper_llr(l1, l2):
    """f 分支 LLR 更新（标量）"""
    return float(f_operation(l1, l2))


def lower_llr(l1, l2, b):
    """g 分支 LLR 更新（标量）"""
    return float(g_operation(l1, l2, b))


def map_channel_llr(llr_ch):
    """
    将信道 LLR 映射到蝶形输出顺序。
    编码器输出 x[k]=u_bf[br[k]]，故 u_bf[j] 对应信道位置 inv_br[j]。
    """
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv_br = np.zeros(N, dtype=int)
    inv_br[br] = np.arange(N)
    return llr_ch[inv_br]


def reorder_llr_to_decoder(llr_ch):
    """兼容接口：映射信道 LLR 至译码器内部顺序"""
    return map_channel_llr(llr_ch)


def _sc_decode_core(llr_internal, frozen_bits):
    """Permuted SC 译码核心（llr_internal 已映射至蝶形顺序）"""
    N = len(llr_internal)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_internal

    for l in [bit_reversed_index(i, n) for i in range(N)]:
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr_internal = map_channel_llr(llr_ch)
    return _sc_decode_core(llr_internal, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，llr 为蝶形顺序）"""
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, offset):
        if len(llr_node) == 1:
            idx = offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return
        half = len(llr_node) // 2
        decode_node(f_operation(llr_node[:half], llr_node[half:]), offset)
        decode_node(
            g_operation(llr_node[:half], llr_node[half:], u_hat[offset : offset + half]),
            offset + half,
        )

    decode_node(np.asarray(llr, dtype=np.float64), 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（供 SCL 使用）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed_index(phi, n)
        start = n - active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        if l < N // 2:
            bit_layer_vec.append([])
        else:
            start_b = n - active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, start_b, -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec
