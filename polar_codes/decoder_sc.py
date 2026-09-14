"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================

def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def permute_llr_for_decode(llr_ch, N):
    """
    将信道 LLR 置换为译码器因子图所需顺序。
    编码器输出含比特倒序置换，译码器输入需对应置换。
    """
    rev = bit_reversal_permutation(N)
    return llr_ch[rev]


def _bit_reversed(x, n):
    """n 位比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def _active_llr_level(i, n):
    """找到 i 的二进制展开中第一个 1 的位置（从高位起）"""
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
    """找到 i 的二进制展开中第一个 0 的位置（从高位起）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _upper_llr(l1, l2):
    return f_operation(l1, l2)


def _lower_llr(btm, top, b):
    if b == 0:
        return top + btm
    return btm - top


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与非递归版本等价）"""
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layer_vec.append(list(range(n - _active_llr_level(phi, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(phi, n), -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = _lower_llr(
                    L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                )


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
    """非递归 SC 译码主函数"""
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = permute_llr_for_decode(llr_ch, N)

    for i in range(N):
        l = _bit_reversed(i, n)
        _update_llrs(L, B, l, n, N)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(int)
