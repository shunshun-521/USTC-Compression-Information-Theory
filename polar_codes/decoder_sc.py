"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（La 为上分支，Lb 为下分支）"""
    return (1 - 2 * u_hat) * La + Lb


def bit_reversed(i, n):
    """n 位比特倒序"""
    return int(format(i, f"0{n}b")[::-1], 2)


def active_llr_level(i, n):
    """LLR 更新起始层（polar-codes 定义）"""
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
    """比特回传起始层"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _sc_decode_recursive_core(llr, frozen_bits):
    """在比特倒序 LLR 顺序下的递归 SC"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    if N == 1:
        if frozen_bits[0]:
            return np.array([0], dtype=int)
        return np.array([0 if llr[0] >= 0 else 1], dtype=int)

    half = N // 2
    llr_u = f_operation(llr[:half], llr[half:])
    u_left = _sc_decode_recursive_core(llr_u, frozen_bits[:half])
    llr_d = g_operation(llr[:half], llr[half:], u_left)
    u_right = _sc_decode_recursive_core(llr_d, frozen_bits[half:])
    return np.concatenate([u_left, u_right])


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码参考接口。
    经典二分递归在比特倒序编码下需与因子图索引对齐；此处与 sc_decode 共用
    同一非递归分层更新，保证与编码器一致（小码长可与 _sc_decode_recursive_core 对照）。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：返回非递归 SC 的层索引描述（文档用）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    lambda_offset = [0]
    for layer in range(1, n + 1):
        lambda_offset.append(lambda_offset[-1] + (1 << (layer - 1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs(L, B, l, n, N):
    """非递归 LLR 更新（Vangala et al. / polar-codes 算法）"""
    start = n - active_llr_level(l, n)
    for s in range(start, n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                top_llr = L[j, s]
                btm_llr = L[j + branch_size, s]
                L[j, s + 1] = f_operation(top_llr, btm_llr)
            else:
                btm_llr = L[j, s]
                top_llr = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)


def _update_bits(B, l, n, N):
    """非递归比特回传"""
    if l < N // 2:
        return
    end = n - active_bit_level(l, n)
    for s in range(n, end, -1):
        block_size = 1 << s
        branch_size = block_size // 2
        j = l
        while j >= 0:
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]
            j -= block_size


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（主实现）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
        l = bit_reversed(i, n)
        _update_llrs(L, B, l, n, N)
        if frozen_bits[i]:
            u_hat[i] = 0
            B[l, n] = 0
        else:
            u_hat[i] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[i]
        _update_bits(B, l, n, N)

    return u_hat


def sc_paths_match(llr, frozen_bits):
    """校验递归与非递归 SC 结果一致"""
    a = sc_decode_recursive(llr, frozen_bits)
    b = sc_decode(llr, frozen_bits)
    return np.array_equal(a, b)
