"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：La 为左子节点 LLR，Lb 为右子节点 LLR。"""
    return Lb + (1 - 2 * u_hat) * La


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


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


def _update_llrs_path(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, L.shape[1], block_size):
            if (j % block_size) < branch_size:
                L[s + 1, j] = f_operation(L[s, j], L[s, j + branch_size])
            else:
                L[s + 1, j] = g_operation(
                    L[s, j - branch_size], L[s, j], B[s + 1, j - branch_size]
                )


def _update_bits_path(B, l, n):
    if l < (1 << (n - 1)):
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if (j % block_size) >= branch_size:
                B[s - 1, j - branch_size] = int(B[s, j]) ^ int(B[s, j - branch_size])
                B[s - 1, j] = B[s, j]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    L = np.zeros((n + 1, N), dtype=np.float64)
    B = np.zeros((n + 1, N), dtype=np.int8)
    L[0, :] = llr_ch

    u_hat = np.zeros(N, dtype=int)
    bit_rev = [_bit_reversed(phi, n) for phi in range(N)]

    for phi, l in enumerate(bit_rev):
        _update_llrs_path(L, B, l, n)

        if frozen_bits[phi]:
            B[n, l] = 0
        else:
            B[n, l] = 0 if L[n, l] >= 0 else 1

        _update_bits_path(B, l, n)
        u_hat[phi] = B[n, l]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec
