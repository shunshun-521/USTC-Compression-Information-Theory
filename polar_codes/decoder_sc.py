"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Vangala 2014 SCD 结构）
"""
import numpy as np


def bit_reversed_index(x, n):
    """单索引比特倒序（与 polarcodes.utils.bit_reversed 一致）"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（upper branch）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（lower branch）：La 为上支路，Lb 为下支路"""
    return (1 - 2 * u_hat) * La + Lb


def active_llr_level(i, n):
    """二进制表示中第一个 1 之前 0 的个数 + 1（上限 n）"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """二进制表示中第一个 0 之前 1 的个数 + 1（上限 n）"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def precompute_sc_indices(N):
    """接口兼容：返回译码顺序等"""
    n = int(np.log2(N))
    decode_order = [bit_reversed_index(i, n) for i in range(N)]
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, [decode_order], [[]]


def _update_llrs(L, B, l, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], top_bit
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    llr_ch: 已与编码端比特倒序对齐（reorder_llr_for_decoder 后）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for l in [bit_reversed_index(i, n) for i in range(N)]:
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """与 sc_decode 等价（保留递归接口）"""
    return sc_decode(llr, frozen_bits)


def get_sc_llr_at_phi(llr, u_hat, phi):
    """给定部分判决 u_hat，返回合成信道 phi 的 LLR（用于 SCL）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.zeros(len(llr), dtype=bool)
    u_hat = np.asarray(u_hat, dtype=int)
    N = len(llr)
    n = int(np.log2(N))
    frozen_set = set()

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr

    decode_order = [bit_reversed_index(i, n) for i in range(N)]
    target = decode_order[phi]

    for idx, l in enumerate(decode_order):
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            if idx < phi:
                B[l, n] = u_hat[l]
            elif l == target:
                return float(L[l, n])
            else:
                B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)
    return float(L[target, n])


def run_sc_self_test():
    from construction import ga_construction
    from encoder import polar_encode
    from channel import (
        bpsk_modulate,
        awgn_channel,
        compute_llr,
        eb_n0_to_sigma,
        reorder_llr_for_decoder,
    )

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = reorder_llr_for_decoder(compute_llr(y, sigma))
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
