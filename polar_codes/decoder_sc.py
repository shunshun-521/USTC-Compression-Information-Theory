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
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed_index(i, n):
    return int(bit_reversal_permutation(1 << n)[i])


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


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))

    if N == 1:
        return np.array([0 if frozen_bits[0] or llr[0] >= 0 else 1])

    half = N // 2
    u_left = sc_decode_recursive(f_operation(llr[:half], llr[half:]), frozen_bits[:half])
    u_right = sc_decode_recursive(g_operation(llr[:half], llr[half:], u_left), frozen_bits[half:])
    return np.concatenate([u_left, u_right])


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（与 active_llr/bit_level 等价）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))

        if l < N // 2:
            bit_layer_vec.append([])
        else:
            stop = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, stop, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（按比特倒序索引顺序译码）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    decode_order = [_bit_reversed_index(i, n) for i in range(N)]
    u_hat = np.zeros(N, dtype=int)

    for l in decode_order:
        start_s = n - _active_llr_level(l, n)
        for s in range(start_s, n):
            block = 1 << (s + 1)
            branch = block >> 1
            for j in range(l, N, block):
                if j % block < branch:
                    top = L[j, s]
                    btm = L[j + branch, s]
                    L[j, s + 1] = f_operation(top, btm)
                else:
                    btm = L[j, s]
                    top = L[j - branch, s]
                    top_bit = B[j - branch, s + 1]
                    L[j, s + 1] = g_operation(top, btm, top_bit)

        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        if l >= N // 2:
            stop_s = n - _active_bit_level(l, n)
            for s in range(n, stop_s, -1):
                block = 1 << s
                branch = block >> 1
                for j in range(l, -1, -block):
                    if j % block >= branch:
                        B[j - branch, s - 1] = B[j, s] ^ B[j - branch, s]
                        B[j, s - 1] = B[j, s]

    return u_hat


def verify_sc_decoders(N=64, num_frames=100, eb_n0_db=10.0, seed=0):
    """在无噪信道上验证 SC 译码无损正确性"""
    from channel import bpsk_modulate, compute_llr
    from construction import ga_construction
    from encoder import polar_encode

    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = 0.01
    rng = np.random.default_rng(seed)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)

        u_rec = sc_decode(llr, frozen_bits)
        if not np.array_equal(u[info_idx], u_rec[info_idx]):
            raise AssertionError("SC 译码信息位错误")
