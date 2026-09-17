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
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def prepare_decoder_llrs(llr_ch):
    """将信道 LLR 重排为译码器所需顺序（与含比特倒序的编码器匹配）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr_dec = np.zeros(N, dtype=np.float64)
    llr_dec[br] = llr_ch
    return llr_dec


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = prepare_decoder_llrs(llr)
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_block(node_llr, offset, length):
        if length == 1:
            idx = offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if node_llr[0] >= 0 else 1
            return

        half = length // 2
        llr_left = f_operation(node_llr[:half], node_llr[half:])
        decode_block(llr_left, offset, half)
        llr_right = g_operation(node_llr[:half], node_llr[half:], u_hat[offset:offset + half])
        decode_block(llr_right, offset + half, half)

    decode_block(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = list(range(n - _active_llr_level(phi, n), n))
        llr_layer_vec.append(llr_layers)

        if phi < N / 2:
            bit_layers = []
        else:
            bit_layers = list(range(n, n - _active_bit_level(phi, n), -1))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    llr = prepare_decoder_llrs(llr_ch)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr
    frozen_set = set(np.where(frozen_bits)[0])

    for phase in [_bit_reversed_index(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(phase, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(phase, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if phase in frozen_set:
            B[phase, n] = 0
        else:
            B[phase, n] = 0 if L[phase, n] >= 0 else 1

        if phase >= N / 2:
            for s in range(n, n - _active_bit_level(phase, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(phase, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
