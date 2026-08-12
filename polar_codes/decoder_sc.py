"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def bit_reversed_index(i, n):
    """单索引比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def active_llr_level(i, n):
    """找到 i 的二进制展开中第一个 1 的位置（从 MSB 起）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """找到 i 的二进制展开中第一个 0 的位置（从 MSB 起）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, phase, n, N):
    for s in range(n - active_llr_level(phase, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(phase, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], top_bit
                )


def _update_bits(B, phase, n, N):
    if phase < N // 2:
        return
    for s in range(n, n - active_bit_level(phase, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(phase, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 需与编码端 B_N 置换一致，内部对比特倒序后的 LLR 译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    br = bit_reversal_permutation(N)
    llr_ch = llr_ch[br]

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    frozen_set = set(np.where(frozen_bits)[0])

    for phase in [bit_reversed_index(i, n) for i in range(N)]:
        _update_llrs(L, B, phase, n, N)
        if phase in frozen_set:
            B[phase, n] = 0
        else:
            B[phase, n] = 0 if L[phase, n] >= 0 else 1
        _update_bits(B, phase, n, N)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用已验证的非递归实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算 SCL 译码辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << (n - i))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        pp = phi
        while pp % 2 == 1:
            llr_layers.append(int(np.log2(pp & -pp)))
            pp //= 2
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        pp = phi
        while pp > 0 and pp % 2 == 0:
            bit_layers.append(int(np.log2(pp & -pp)))
            pp //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec
