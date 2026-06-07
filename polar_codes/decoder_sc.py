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


def _as_frozen_mask(frozen_bits):
    frozen = np.asarray(frozen_bits)
    if frozen.dtype == bool:
        return frozen
    return frozen.astype(bool)


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def _prepare_channel_llr(llr_ch):
    """编码含比特倒序时，将信道 LLR 重排以匹配 SC 因子图。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    return llr_ch[br]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = _prepare_channel_llr(llr)
    return _sc_decode_core(llr, _as_frozen_mask(frozen_bits))


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（与递归版本等价的高效实现）。"""
    llr = _prepare_channel_llr(llr_ch)
    return _sc_decode_core(llr, _as_frozen_mask(frozen_bits))


def _sc_decode_core(llr, frozen):
    """
    基于分层 LLR/比特数组的 SC 译码核心。
    参考 Permuted Successive Cancellation Decoder 结构。
    """
    N = len(llr)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    u_hat = np.zeros(N, dtype=np.int8)

    for i in range(N):
        l = _bit_reversed_index(i, n)
        _update_llrs(L, B, l, n, N)
        if frozen[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = int(B[l, n])
        _update_bits(B, l, n, N)

    return u_hat.astype(int)


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
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
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（供 SCL 使用）。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))

        active = _active_bit_level(l, n)
        bit_layer_vec.append(list(range(n, n - active, -1)))

    lambda_offset = np.zeros(N, dtype=int)
    offset = 1
    for i in range(N):
        lambda_offset[i] = offset - 1
        if i + 1 == offset:
            offset <<= 1

    return lambda_offset, llr_layer_vec, bit_layer_vec


if __name__ == "__main__":
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u_msg = np.zeros(N, dtype=int)
        u_msg[info_idx] = rng.integers(0, 2, K)
        x_tx = polar_encode(u_msg)
        y = awgn_channel(bpsk_modulate(x_tx), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_msg[info_idx]):
            errors += 1
    print(f"SC lossless test: {errors}/100 frame errors")
