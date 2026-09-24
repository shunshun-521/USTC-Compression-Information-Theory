"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """找到 i 的二进制表示中第一个 1 的位置（从高位起）"""
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
    """找到 i 的二进制表示中第一个 0 的位置（从高位起）"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_reversed(i, n):
    """比特倒序索引"""
    result = 0
    for _ in range(n):
        result = (result << 1) | (i & 1)
        i >>= 1
    return result


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回：lambda_offset, llr_layer_vec, bit_layer_vec
    """
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = 1 << (i - 1)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        layers_llr = list(range(_active_llr_level(l, n) - 1, n))
        layers_bit = list(range(n, n - _active_bit_level(l, n), -1))
        llr_layer_vec.append(layers_llr)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_core(llr, frozen_bits):
    """
    非递归 SC 译码核心（参考 mcba1n/polar-codes SCD 实现）。
    llr: 长度 N，与信道码字 x 的自然顺序对应。
    frozen_bits: 长度 N，True 表示冻结位。
    返回：长度 N 的估计源序列 u_hat（自然顺序）。
    """
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    for phi in range(N):
        l = _bit_reversed(phi, n)

        # 更新 LLR 树
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s],
                        int(B[j - branch_size, s + 1])
                    )

        # 硬判决
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        # 比特回传
        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr = np.asarray(llr_ch, dtype=np.float64)[br]
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, fbits, offset):
        n = len(llr_node)
        if n == 1:
            u_hat[offset] = 0 if fbits[0] or llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, fbits[:half], offset)
        u_left = u_hat[offset:offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, fbits[half:], offset + half)

    decode_node(llr, frozen_bits, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 为自然顺序（对应 polar_encode 输出的码字顺序）。
    polar_encode 在编码末尾做比特倒序置换，因此译码前需对 LLR 做相同置换。
    """
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr = np.asarray(llr_ch, dtype=np.float64)[br]
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return _sc_decode_core(llr, frozen_bits)
