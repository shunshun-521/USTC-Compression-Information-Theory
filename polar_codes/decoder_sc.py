"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def _sign_pm(x):
    """符号函数，零值按 +1 处理。"""
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return _sign_pm(La) * _sign_pm(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(btm_llr, top_llr, u_hat):
    """
    g 运算：g(btm, top, u) = btm + (1 - 2*u) * top
    """
    return btm_llr + (1 - 2 * u_hat) * top_llr


def active_llr_level(i, n):
    """找到 i 的二进制表示中第一个 1 的位置（从高位起）。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """找到 i 的二进制表示中第一个 0 的位置（从高位起）。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_rev(x, n):
    """单值比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(np.log2(N))
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi in range(N):
        for s in range(n - active_llr_level(phi, n), n):
            llr_layer_vec[phi].append(s)
        if phi >= N // 2:
            for s in range(n, n - active_bit_level(phi, n), -1):
                bit_layer_vec[phi].append(s)

    return llr_layer_vec, bit_layer_vec


def _update_llrs(L, B, l, n):
    """更新 LLR 树。"""
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j, s],
                    L[j - branch_size, s],
                    B[j - branch_size, s + 1],
                )


def _update_bits(B, l, n, N):
    """比特回传。"""
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)

    if N == 1:
        if frozen_bits[0]:
            return np.array([0], dtype=int)
        return np.array([0 if llr[0] >= 0 else 1], dtype=int)

    half = N // 2
    llr_left = f_operation(llr[:half], llr[half:])
    u_left = sc_decode_recursive(llr_left, frozen_bits[:half])
    llr_right = g_operation(llr[half:], llr[:half], u_left)
    u_right = sc_decode_recursive(llr_right, frozen_bits[half:])
    return np.concatenate([u_left, u_right])


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Permuted SC，与蝶形+比特倒序编码配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    brp = bit_reversal_permutation(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch[brp]

    u_hat = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits)[0])
    decode_order = [_bit_rev(i, n) for i in range(N)]

    for l in decode_order:
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = int(B[l, n])
        _update_bits(B, l, n, N)

    return u_hat
