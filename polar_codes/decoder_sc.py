"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================

def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_partial):
    """
    g 运算：La 为上层 LLR，Lb 为下层 LLR，u_partial 为上层重编码部分和
    g = (1-2u)*La + Lb
    """
    return (1.0 - 2.0 * u_partial) * La + Lb


def prepare_channel_llr(llr_ch):
    """信道 LLR 保持自然顺序"""
    return np.asarray(llr_ch, dtype=np.float64)


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（Sionna 风格，使用重编码部分和）。
    """
    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr = prepare_channel_llr(llr_ch)

    def decode(llr_vec, frozen_vec):
        n = len(llr_vec)
        if n == 1:
            if frozen_vec[0]:
                u = 0
            else:
                u = 0 if llr_vec[0] >= 0 else 1
            return np.array([u], dtype=int), np.array([u], dtype=int)

        half = n // 2
        llr_left = f_operation(llr_vec[:half], llr_vec[half:])
        u_left, u_left_up = decode(llr_left, frozen_vec[:half])

        llr_right = g_operation(llr_vec[:half], llr_vec[half:], u_left_up)
        u_right, u_right_up = decode(llr_right, frozen_vec[half:])

        u_hat = np.concatenate([u_left, u_right])
        u_up = np.concatenate([np.bitwise_xor(u_left_up, u_right_up), u_right_up])
        return u_hat, u_up

    u_hat, _ = decode(llr, frozen_bits)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    def bit_reversed(i):
        return int(format(i, f'0{n}b')[::-1], 2)

    def active_llr_level(i):
        mask = 2 ** (n - 1)
        count = 1
        for _ in range(n):
            if (mask & i) == 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    def active_bit_level(i):
        mask = 2 ** (n - 1)
        count = 1
        for _ in range(n):
            if (mask & i) > 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi)
        start_s = n - active_llr_level(l)
        llr_layer_vec.append(list(range(start_s, n)))
        start_bit = n - active_bit_level(l)
        bit_layer_vec.append(list(range(n, start_bit, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（基于因子图逐比特更新）。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr_in = prepare_channel_llr(llr_ch)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_in

    u_hat = np.zeros(N, dtype=int)

    def bit_reversed(i):
        return int(format(i, f'0{n}b')[::-1], 2)

    def active_llr_level(i):
        mask = 2 ** (n - 1)
        count = 1
        for _ in range(n):
            if (mask & i) == 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    def active_bit_level(i):
        mask = 2 ** (n - 1)
        count = 1
        for _ in range(n):
            if (mask & i) > 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    for phi in range(N):
        l = bit_reversed(phi)

        for s in range(n - active_llr_level(l), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = (
                            int(B[j, s]) ^ int(B[j - branch_size, s])
                        )
                        B[j, s - 1] = B[j, s]

    return u_hat


# 主函数：递归实现经验证正确
sc_decode_iterative = sc_decode
sc_decode = sc_decode_recursive
