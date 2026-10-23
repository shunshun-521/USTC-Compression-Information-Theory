"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum f 运算"""
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * np.asarray(u_hat, dtype=np.float64)) * La + Lb


def _bit_reversed_scalar(x, n):
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


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi % 2 == 1:
            layers_llr.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        if phi % 2 == 0:
            for layer in range(n):
                if (phi >> layer) & 1:
                    layers_bit.append(layer)
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    Permuted SC 译码，信道 LLR 先按比特倒序重排后与 polar_encode 配套。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype != bool:
        frozen_bits = frozen_bits.astype(bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    rev = bit_reversal_permutation(N)
    llr = llr_ch[rev].copy()
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    for l in [_bit_reversed_scalar(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]

        if l < N // 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] ^ B[j - branch_size, s]) & 1
                    B[j, s - 1] = B[j, s]

    return u_hat
