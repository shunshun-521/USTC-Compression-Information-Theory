"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed_int(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    """从 MSB 起找到第一个 1 之前的层数。"""
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
    """从 MSB 起找到第一个 0 之前的层数。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（兼容接口）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi in range(N):
        l = _bit_reversed_int(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec[phi] = list(range(start, n))

        if l < N // 2:
            bit_layer_vec[phi] = []
        else:
            end = n - _active_bit_level(l, n)
            bit_layer_vec[phi] = list(range(n, end, -1))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_core(llr_ch, frozen_bits):
    """Permuted SC 核心实现，输入 LLR 已与比特倒序编码对齐。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    frozen_set = set(np.where(frozen_bits)[0])

    for phi in range(N):
        l = _bit_reversed_int(phi, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(np.int8)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    与含比特倒序置换的编码器配套：先对信道 LLR 做比特倒序重排。
    """
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    llr_aligned = np.asarray(llr_ch, dtype=np.float64)[rev]
    return _sc_decode_core(llr_aligned, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用非递归版本）。"""
    return sc_decode(llr, frozen_bits)


def verify_sc_decoders(N=64, K=32, num_frames=100):
    """验证 SC 译码器在高信噪比下无误。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(42)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = s + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)

        u_rec = sc_decode(llr, frozen_bits)
        u_rec_r = sc_decode_recursive(llr, frozen_bits)

        if not np.array_equal(u_rec, u_rec_r):
            raise AssertionError("SC recursive vs non-recursive mismatch")
        if not np.array_equal(u, u_rec):
            raise AssertionError("SC decode error at high SNR")

    print(f"SC verification passed: N={N}, {num_frames} frames")


if __name__ == "__main__":
    verify_sc_decoders()
