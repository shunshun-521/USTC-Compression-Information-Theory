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
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb。"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    """从最高位起，第一个 0 之前 1 的个数 + 1（Permuted SC）。"""
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
    """从最高位起，第一个 0 之前 1 的个数 + 1（比特回传）。"""
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
    """
    递归 SC 译码（跨半块步长合并 LLR）。
    llr 需为 llr_ch[bit_rev] 顺序（与蝶形编码对齐）。
    """
    llr = list(np.asarray(llr, dtype=np.float64))
    frozen_bits = list(np.asarray(frozen_bits, dtype=bool))
    n = len(llr)
    decisions = [0] * n

    def sc(llr_node, indices):
        if len(indices) == 1:
            i = indices[0]
            decisions[i] = 0 if frozen_bits[i] or llr_node[0] >= 0 else 1
            return [decisions[i]]

        half = len(indices) // 2
        left_llr = [
            float(f_operation(llr_node[i], llr_node[i + half])) for i in range(half)
        ]
        enc_left = sc(left_llr, indices[:half])
        right_llr = [
            float(g_operation(llr_node[i], llr_node[i + half], enc_left[i]))
            for i in range(half)
        ]
        enc_right = sc(right_llr, indices[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    sc(llr, list(range(n)))
    return np.array(decisions, dtype=int)


def precompute_sc_indices(N):
    """预计算非递归 SC 的层索引表。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        act_llr = _active_llr_level(phi, n)
        act_bit = _active_bit_level(phi, n)
        llr_layer_vec.append(list(range(n - act_llr, n)))
        bit_layer_vec.append(list(range(n - act_bit, n)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs_perm(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits_perm(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SC 译码（主实现）。
    llr_ch 与 polar_encode 输出比特顺序一致。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    br = bit_reversal_permutation(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch[br]

    for l in [_bit_reversed(i, n) for i in range(N)]:
        _update_llrs_perm(L, B, l, n, N)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits_perm(B, l, n, N)

    return B[:, n].astype(int)


def sc_decode_nonrecursive_table(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)
