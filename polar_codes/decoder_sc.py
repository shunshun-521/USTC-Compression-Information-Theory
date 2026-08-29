"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def _sign_ms(x):
    """min-sum 用符号函数，0 映射为 +1"""
    return np.where(np.asarray(x) >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
  支持向量化
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return _sign_ms(La) * _sign_ms(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算（与因子图下支路一致）：
    u=0 时 La+Lb，u=1 时 La-Lb
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def _bit_reversed_index(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if (x >> i) & 1:
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    """LLR 更新起始层（从 MSB 起首个 0）"""
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
    """比特回传起始层（从 MSB 起首个 1）"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    信道 LLR 先做比特倒序以匹配编码端置换。
    """
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(math.log2(N))
    rev = bit_reversal_permutation(N)
    llr = llr[rev]

    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = _bit_reversed_index(i, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            u_hat[l] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
        B[l, n] = u_hat[l]

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = (
                            B[j, s] ^ B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（与 mcba1n 风格算法对应）。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))

        bit_start = n - _active_bit_level(l, n)
        bit_layer_vec.append(list(range(n, bit_start, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 先做比特倒序以匹配编码端置换。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    rev = bit_reversal_permutation(N)
    llr_ch = llr_ch[rev]

    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = _bit_reversed_index(phi, n)

        for s in llr_layer_vec[phi]:
            block_size = lambda_offset[s + 1]
            branch_size = lambda_offset[s]
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            u_hat[l] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
        B[l, n] = u_hat[l]

        if l >= N // 2:
            for s in bit_layer_vec[phi]:
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = (
                            B[j, s] ^ B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    return u_hat
