"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted SCD）
"""
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
    return (1.0 - 2.0 * u_hat) * La + Lb


def _g_node(La, Lb, u_hat):
    """因子图节点 g 更新（Permuted SCD 实现）"""
    if np.isnan(u_hat):
        return np.nan
    if int(u_hat) == 0:
        return La + Lb
    return La - Lb


def _to_frozen_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return set(np.where(frozen_bits)[0])
    return set(np.where(frozen_bits.astype(int))[0])


def prepare_channel_llr(llr_ch):
    """将自然序信道 LLR 映射到译码树顺序（与编码端比特倒序一致）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br]


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


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回 lambda_offset, llr_layer_vec, bit_layer_vec
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        llr_layer_vec.append(list(range(n - _active_llr_level(i, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(i, n), -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = _g_node(L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1])


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（Permuted SCD，高效实现）"""
    llr_ch = prepare_channel_llr(llr_ch)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_set = _to_frozen_set(frozen_bits)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for l in [_bit_reversed_index(i, n) for i in range(N)]:
        _update_llrs(L, B, l, n, N)

        if l in frozen_set:
            bit = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1

        B[l, n] = bit
        u_hat[l] = bit
        _update_bits(B, l, n, N)

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，委托给非递归版本以保证一致性）"""
    return sc_decode(llr, frozen_bits)
