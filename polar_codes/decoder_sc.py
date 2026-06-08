"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，按比特倒序索引译码）
"""
import math
import numpy as np
from encoder import bit_reversed

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """译码第 i 个比特时需更新的 LLR 层数（首个 1 的位置）"""
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
    """比特回传时需更新的层数（首个 0 的位置）"""
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
    """递归 SC 译码（按自然信道索引顺序）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_block(llr_node, frozen_block):
        n = len(llr_node)
        if n == 1:
            if frozen_block[0]:
                return np.array([0], dtype=int)
            return np.array([0 if llr_node[0] >= 0 else 1], dtype=int)

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left = decode_block(llr_left, frozen_block[:half])
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        u_right = decode_block(llr_right, frozen_block[half:])
        return np.concatenate([u_left, u_right])

    return decode_block(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    按比特倒序索引顺序处理各合成信道（与标准极化码 SC 实现一致）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = bit_reversed(i, n)

        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            half = block // 2
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
                else:
                    top_bit = B[j - half, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - half, s], L[j, s], top_bit
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]

        if l < N // 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block = 1 << s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    B[j - half, s - 1] = B[j, s] ^ B[j - half, s]
                    B[j, s - 1] = B[j, s]

    return u_hat
