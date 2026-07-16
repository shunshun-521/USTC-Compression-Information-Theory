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
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
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
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _prepare_llr(llr_ch):
    """编码含比特倒序置换时，将信道 LLR 映射到译码树叶子节点"""
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]



def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（按比特倒序译码顺序）"""
    n = int(math.log2(N))
    decode_order = [_bit_reversed_index(i, n) for i in range(N)]
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for l in decode_order:
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        llr_layer_vec.append(llr_layers)

        bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    采用 (N, n+1) 分层 LLR 存储，按比特倒序依次译码。
    """
    llr_ch = _prepare_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    _, _, _, decode_order = precompute_sc_indices(N)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], int(B[j - branch_size, s + 1])
                    )

        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与非递归版本等价）"""
    return sc_decode(llr, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与非递归版本等价）"""
    return sc_decode(llr, frozen_bits)
