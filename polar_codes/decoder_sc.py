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
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算（btm, top 顺序）：btm + top (u=0) 或 btm - top (u=1)
    等价于 (1 - 2*u_hat) * top + btm
    """
    btm, top = La, Lb
    return btm + top if u_hat == 0 else btm - top


def _bit_reversed_index(x, n):
    """单索引比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def _active_llr_level(i, n):
    """LLR 更新起始层（首个 1 之前的 0 个数）。"""
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
    """比特回传起始层（首个 0 之前的 1 个数）。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def prepare_llr_for_decode(llr_ch, N):
    """将信道 LLR 做比特倒序置换，与编码器输出顺序对齐。"""
    rev = bit_reversal_permutation(N)
    return llr_ch[rev]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    frozen_bits: True/1 表示冻结位
    """
    llr = prepare_llr_for_decode(np.asarray(llr, dtype=np.float64), len(llr))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def decode_block(llr_vec, depth, offset):
        m = len(llr_vec)
        if m == 1:
            if frozen_bits[offset]:
                u_hat[offset] = 0
            else:
                u_hat[offset] = 0 if llr_vec[0] >= 0 else 1
            return
        half = m // 2
        llr_left = f_operation(llr_vec[:half], llr_vec[half:])
        decode_block(llr_left, depth - 1, offset)
        u_left = u_hat[offset:offset + half].copy()
        llr_right = np.array([
            g_operation(llr_vec[i + half], llr_vec[i], u_left[i])
            for i in range(half)
        ])
        decode_block(llr_right, depth - 1, offset + half)

    decode_block(llr, n, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（Tal-Vardy 风格索引）。
    """
    n = int(math.log2(N))
    lambda_offset = np.array([2 ** i for i in range(n + 1)], dtype=int)

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))

        if l < N / 2:
            bit_layer_vec.append([])
        else:
            start_b = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, start_b, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    llr_ch: 信道自然顺序 LLR；内部自动做比特倒序。
    frozen_bits: 长度 N，1/True 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    llr = prepare_llr_for_decode(llr_ch, N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = _bit_reversed_index(i, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return u_hat
