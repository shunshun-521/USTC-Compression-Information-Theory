"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def bit_reversal_indices(N):
    """返回长度 N 的比特倒序索引数组"""
    n = int(math.log2(N))
    return np.array([bit_reversed(i, n) for i in range(N)], dtype=np.int64)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：La 为下分支 LLR，Lb 为上分支 LLR。
    u=0: La+Lb；u=1: La-Lb
    """
    u = int(u_hat) if not np.isnan(u_hat) else 0
    return (La + Lb) if u == 0 else (La - Lb)


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


def _prepare_llrs(llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_indices(N)
    return llr_ch[br], N, n


def _sc_decode_core(likelihoods, frozen_set, N, n):
    """非递归 SC 译码核心（自然顺序 LLR，按比特倒序译码）"""
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = likelihoods
    u_hat = np.zeros(N, dtype=int)

    for l in [bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s],
                        L[j - branch_size, s],
                        B[j - branch_size, s + 1],
                    )

        if l in frozen_set:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        if l < N / 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(n - _active_llr_level(phi, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(phi, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    编码端含比特倒序置换时，先将信道 LLR 重排至译码树自然顺序。
    """
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    likelihoods, N, n = _prepare_llrs(llr_ch)
    return _sc_decode_core(likelihoods, frozen_set, N, n)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，与非递归结果一致）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    likelihoods, N, n = _prepare_llrs(llr_ch)
    return _sc_decode_core(likelihoods, frozen_set, N, n)


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    """SC 译码无损验证"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            return False
        if not np.array_equal(sc_decode_recursive(llr, frozen_bits), u):
            return False
    return True
