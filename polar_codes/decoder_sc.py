"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversed


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """LLR 更新起始层（首个 1 的位置，从 MSB 计数）"""
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
    """比特回传起始层（首个 0 的位置，从 MSB 计数）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_block(llr_node, frozen_block, offset):
        n = len(llr_node)
        if n == 1:
            if frozen_block[0]:
                u_hat[offset] = 0
            else:
                u_hat[offset] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_block(llr_left, frozen_block[:half], offset)
        u_left = u_hat[offset : offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_block(llr_right, frozen_block[half:], offset + half)

    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    decode_block(np.asarray(llr, dtype=np.float64), frozen_bits, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（按比特倒序相位，参考 Vangala et al. 2014）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    for i in range(N):
        phase = bit_reversed(i, n)

        start_layer = n - _active_llr_level(phase, n)
        for s in range(start_layer, n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(phase, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )

        if frozen_bits[phase]:
            B[phase, n] = 0
        else:
            B[phase, n] = 0 if L[phase, n] >= 0 else 1

        if phase < N // 2:
            continue

        end_layer = n - _active_bit_level(phase, n)
        for s in range(n, end_layer, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(phase, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
