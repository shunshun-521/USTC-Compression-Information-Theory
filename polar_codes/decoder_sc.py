"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    La 为上层（top）分支，Lb 为下层（bottom）分支。
    """
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
    """整数比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def _active_llr_level(i, n):
    """llr_layer_vec 辅助：从高位起第一个 1 之前连续 0 的个数"""
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
    """bit_layer_vec 辅助：从高位起第一个 0 之前连续 1 的个数"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l_idx = _bit_reversed_index(phi, n)
        llr_layers = list(range(n - _active_llr_level(l_idx, n), n))
        llr_layer_vec.append(llr_layers)

        bit_layers = list(range(n, n - _active_bit_level(l_idx, n), -1))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _prepare_llr(llr_ch):
    """编码含比特倒序置换，信道 LLR 需对应重排"""
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，按比特倒序逐位译码）。
    """
    N = len(llr)
    n = int(math.log2(N))
    llr = _prepare_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    u_hat = np.zeros(N, dtype=int)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr

    def update_llrs(l_idx):
        for s in range(n - _active_llr_level(l_idx, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l_idx, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )

    def update_bits(l_idx):
        if l_idx < N // 2:
            return
        for s in range(n, n - _active_bit_level(l_idx, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                    B[j, s - 1] = B[j, s]

    def decode_bit(phi):
        if phi >= N:
            return
        l_idx = _bit_reversed_index(phi, n)
        update_llrs(l_idx)
        if l_idx in frozen_set:
            u_hat[l_idx] = 0
        else:
            u_hat[l_idx] = 0 if L[l_idx, n] >= 0 else 1
        B[l_idx, n] = u_hat[l_idx]
        update_bits(l_idx)
        decode_bit(phi + 1)

    decode_bit(0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（高效实现）。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    llr = _prepare_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l_idx = _bit_reversed_index(phi, n)

        for s in range(n - _active_llr_level(l_idx, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l_idx, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )

        if l_idx in frozen_set:
            u_hat[l_idx] = 0
        else:
            u_hat[l_idx] = 0 if L[l_idx, n] >= 0 else 1
        B[l_idx, n] = u_hat[l_idx]

        if l_idx >= N // 2:
            for s in range(n, n - _active_bit_level(l_idx, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l_idx, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                        B[j, s - 1] = B[j, s]

    return u_hat
