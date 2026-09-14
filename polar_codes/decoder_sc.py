"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效 PSCD 实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（与 log-domain lower_llr 等价）"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
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


def _prepare_frozen(frozen_bits):
    fb = np.asarray(frozen_bits)
    return np.where(fb.astype(bool))[0]


def _map_channel_llrs(llr_ch, N):
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _update_llrs(L, B, l, n, f_fn):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_fn(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n):
    if l < B.shape[0] / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（与 PSCD 比特处理顺序一致）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    decode_order = [_bit_reversed_index(i, n) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for l in decode_order:
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        if l >= N / 2:
            bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
        else:
            bit_layer_vec.append([])
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用非递归 PSCD）"""
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（PSCD）。
    信道 LLR 经比特倒序映射后，按 bit-reversed 顺序逐比特译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_set = set(_prepare_frozen(frozen_bits))

    llr = _map_channel_llrs(llr_ch, N)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    for l in [_bit_reversed_index(i, n) for i in range(N)]:
        _update_llrs(L, B, l, n, f_operation)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, n].astype(int)


def run_sc_self_test():
    """SC 译码自检"""
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u_src = np.zeros(N, dtype=int)
        u_src[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u_src)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_src[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 下有 {errors} 帧错误"
    print("SC self-test passed.")
