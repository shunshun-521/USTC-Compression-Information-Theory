"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
    """单整数比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def _active_llr_level(i, n):
    """找二进制表示中第一个 1 的位置（从高位起）"""
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
    """找二进制表示中第一个 0 的位置（从高位起）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
  返回 bit_reversed 译码顺序及活跃层信息。
    """
    n = int(math.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = [_active_llr_level(l, n) for l in decode_order]
    bit_layer_vec = [_active_bit_level(l, n) for l in decode_order]
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    对信道 LLR 做比特倒序后译码。
    """
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    rev = bit_reversal_permutation(N)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def _rec(llr_node, offset):
        n = len(llr_node)
        if n == 1:
            if frozen_bits[offset]:
                return np.array([0])
            return np.array([0 if llr_node[0] >= 0 else 1])
        half = n // 2
        llr_u = f_operation(llr_node[:half], llr_node[half:])
        u_left = _rec(llr_u, offset)
        llr_d = g_operation(llr_node[:half], llr_node[half:], u_left)
        u_right = _rec(llr_d, offset + half)
        return np.concatenate([u_left, u_right])

    return _rec(llr[rev], 0)


def _sc_decode_core(llr_ch, frozen_set, N):
    """
    非递归 SC 译码核心（树形 L/B 数组）。
    llr_ch 已按比特倒序排列。
    """
    n = int(math.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N / 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    对信道 LLR 做比特倒序后译码（与编码器比特倒序对应）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
    return _sc_decode_core(llr_ch[rev], frozen_set, N)
