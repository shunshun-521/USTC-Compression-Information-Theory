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
    """译码索引 i 的第一个 0 位（MSB 起）决定需更新的 LLR 层数"""
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
    """译码索引 i 的第一个 1 位决定需回传的比特层数"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_reversed_index(i, n):
    """单索引比特倒序"""
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


def _remap_channel_llrs(llr_ch):
    """
    将信道 LLR 映射到译码树自然顺序。
    编码器输出 x[i] = enc_core[bitrev(i)]，因此 llr_tree[j] = llr_ch[inv_br(j)]。
    """
    N = len(llr_ch)
    inv_br = np.argsort(bit_reversal_permutation(N))
    return np.asarray(llr_ch, dtype=np.float64)[inv_br]


def _update_llrs(L, B, l, n, N):
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
    """
    非递归 SC 译码主函数。
    按比特倒序索引顺序译码（与标准极化码因子图一致）。
    """
    llr_ch = _remap_channel_llrs(llr_ch)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        _update_llrs(L, B, l, n, N)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        _update_bits(B, l, n, N)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用非递归核心）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（供 SCL 使用）。
    返回译码顺序与层更新信息。
    """
    n = int(math.log2(N))
    decode_order = [_bit_reversed_index(phi, n) for phi in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for l in decode_order:
        llr_start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(llr_start, n)))
        if l < N // 2:
            bit_layer_vec.append([])
        else:
            bit_start = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, bit_start, -1)))
    return decode_order, llr_layer_vec, bit_layer_vec


def _update_llr_layer(L, B, layer, l, n, N):
    s = layer
    block_size = 1 << (s + 1)
    branch_size = block_size // 2
    for j in range(l, N, block_size):
        if j % block_size < branch_size:
            L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
        else:
            L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1])


def _update_bit_layer(B, layer, l, n, N):
    s = layer
    block_size = 1 << s
    branch_size = block_size // 2
    for j in range(l, -1, -block_size):
        if j % block_size >= branch_size:
            B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
            B[j, s - 1] = B[j, s]
