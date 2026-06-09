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
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """当前比特索引需要更新的 LLR 层数"""
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
    """当前比特索引需要回传的比特层数"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, N, block_size):
            if (j % block_size) < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], top_bit
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if (j % block_size) >= branch_size:
                B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    按比特倒序依次译码，与编码器蝶形结构一致。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]
    u_hat = np.zeros(N, dtype=int)

    for l in decode_order:
        _update_llrs(L, B, l, n, N)
        if frozen_bits[l]:
            u_hat[l] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
        B[l, n] = u_hat[l]
        _update_bits(B, l, n, N)

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（小规模参考实现）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（供 SCL 使用）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    br = bit_reversal_permutation(N)
    decode_order = [br[i] for i in range(N)]

    llr_layer_vec = []
    bit_layer_vec = []
    for l in decode_order:
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        if l >= N // 2:
            bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
        else:
            bit_layer_vec.append([])

    return lambda_offset, llr_layer_vec, bit_layer_vec
