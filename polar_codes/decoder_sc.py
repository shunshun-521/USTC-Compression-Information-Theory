"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return int(bin(i)[2:].zfill(n)[::-1], 2)


def _bit_reversal_array(N):
    n = int(math.log2(N))
    return np.array([_bit_reversed(i, n) for i in range(N)])


def _active_llr_level(i, n):
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
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _prepare_llr(llr_ch):
    """将信道 LLR 转换为译码器内部顺序（与编码端比特倒序对应）"""
    N = len(llr_ch)
    return np.asarray(llr_ch, dtype=np.float64)[_bit_reversal_array(N)]


def _update_llr_recursive(L, B, l, s, n):
    """递归更新 LLR（参考实现，按层递归）"""
    if s >= n:
        return
    block_size = 2 ** (s + 1)
    branch_size = block_size // 2
    for j in range(l, len(L), block_size):
        if j % block_size < branch_size:
            L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
        else:
            L[j, s + 1] = g_operation(
                L[j - branch_size, s],
                L[j, s],
                B[j - branch_size, s + 1],
            )
    _update_llr_recursive(L, B, l, s + 1, n)


def _update_bits_recursive(B, l, s, start_b, n):
    """递归更新部分和（参考实现，按层递归）"""
    if s <= start_b or l < len(B) // 2:
        return
    block_size = 2 ** s
    branch_size = block_size // 2
    for j in range(l, -1, -block_size):
        if j % block_size >= branch_size:
            B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
            B[j, s - 1] = B[j, s]
    _update_bits_recursive(B, l, s - 1, start_b, n)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回 decode_order, llr_start_stage, bit_start_stage, n
    """
    n = int(math.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    llr_start_stage = [_active_llr_level(l, n) for l in decode_order]
    bit_start_stage = [_active_bit_level(l, n) for l in decode_order]
    return decode_order, llr_start_stage, bit_start_stage, n


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，与非递归结果一致）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    decode_order, llr_start, bit_start, _ = precompute_sc_indices(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = _prepare_llr(llr_ch)
    u_hat = np.zeros(N, dtype=int)

    for idx, l in enumerate(decode_order):
        start_s = n - llr_start[idx]
        _update_llr_recursive(L, B, l, start_s, n)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]

        if l >= N // 2:
            start_b = n - bit_start[idx]
            _update_bits_recursive(B, l, n, start_b, n)

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（高效实现）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    decode_order, llr_start, bit_start, n = precompute_sc_indices(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = _prepare_llr(llr_ch)

    for idx, l in enumerate(decode_order):
        start_s = n - llr_start[idx]
        for s in range(start_s, n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1]
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            start_b = n - bit_start[idx]
            for s in range(n, start_b, -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n]
