"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
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
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        for i in range(half):
            decode_node(llr_left[i : i + 1], bit_offset + i)

        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i : i + 1], bit_offset + half + i)

    decode_node(llr, 0)
    return u_hat


def _precompute_sc_tables(N):
    """预计算非递归 SC 译码的层更新模式"""
    n = int(math.log2(N))
    llr_layers = []
    bit_layers = []
    for phi in range(N):
        llr_l = []
        psi = phi
        for layer in range(n):
            if (psi & 1) == 0:
                llr_l.append(layer)
                psi >>= 1
            else:
                break
        llr_layers.append(llr_l)

        bit_l = []
        for layer in range(n):
            if (phi >> layer) & 1:
                bit_l.append(layer)
        bit_layers.append(bit_l)

    return llr_layers, bit_layers


def _active_llr_level(i, n):
    """从最高位开始统计连续 0 的个数"""
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
    """从最高位开始统计连续 1 的个数"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（置换顺序，min-sum 近似）"""
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    br = bit_reversal_permutation(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = np.asarray(llr_ch, dtype=np.float64)

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        leaf = br[phi]

        for s in range(n - _active_llr_level(leaf, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(leaf, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if frozen_bits[phi]:
            u_hat[phi] = 0
            B[leaf, n] = 0
        else:
            bit = 0 if L[leaf, n] >= 0 else 1
            u_hat[phi] = bit
            B[leaf, n] = bit

        if leaf >= N // 2:
            for s in range(n, n - _active_bit_level(leaf, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(leaf, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return u_hat
