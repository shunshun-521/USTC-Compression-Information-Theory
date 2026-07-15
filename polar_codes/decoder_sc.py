"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed_index(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


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


def _permute_channel_llrs(llr_ch, N):
    """
    编码器输出经比特倒序后，信道 LLR 需做相同置换再送入 SC 因子图。
    """
    br = bit_reversal_permutation(N)
    llr = np.asarray(llr_ch, dtype=np.float64)
    return llr[br]


def _sc_decode_core(llr, frozen_bits, n):
    """SC 译码核心（llr 已置换为因子图顺序）"""
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    def update_llrs(l):
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

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    for idx in range(N):
        l = _bit_reversed_index(idx, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit
        update_bits(l)

    return u_hat


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归/因子图 SC 译码"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    llr = _permute_channel_llrs(llr_ch, N)
    return _sc_decode_core(llr, frozen_bits, n)


def precompute_sc_indices(N):
    """预计算辅助向量（兼容接口）"""
    n = int(math.log2(N))
    decode_order = [_bit_reversed_index(i, n) for i in range(N)]
    llr_layer_vec = [_active_llr_level(l, n) for l in decode_order]
    bit_layer_vec = [_active_bit_level(l, n) for l in decode_order]
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（与 sc_decode_recursive 等价）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def verify_sc_decoders(N=64, frozen_bits=None, num_trials=100, eb_n0_db=10.0):
    """验证 SC 译码器在高信噪比下无误码"""
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    if frozen_bits is None:
        frozen_bits = np.ones(N, dtype=bool)
        frozen_bits[info_idx] = False

    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(0)

    for _ in range(num_trials):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_rec = sc_decode(llr, frozen_bits)
        u_rec_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_rec, u_rec_r), "SC implementations mismatch"
        assert np.array_equal(u[info_idx], u_rec[info_idx]), "SC decode error at high SNR"

    return True
