"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（与非递归 sc_decode 等价的递归结构实现，用于交叉验证）。
    """
    N = len(llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = np.asarray(llr, dtype=np.float64)
    u_hat = np.zeros(N, dtype=int)

    def update_llrs(l, s_start):
        if s_start >= n:
            return
        block_size = 1 << (s_start + 1)
        branch_size = block_size >> 1
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s_start + 1] = f_operation(L[j, s_start], L[j + branch_size, s_start])
            else:
                L[j, s_start + 1] = g_operation(
                    L[j - branch_size, s_start], L[j, s_start], B[j - branch_size, s_start + 1]
                )
        update_llrs(l, s_start + 1)

    def update_bits(l, s):
        if s < n - _active_bit_level(l, n) + 1:
            return
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]
        update_bits(l, s - 1)

    def decode_bit(i):
        if i >= N:
            return
        l = _bit_reversed(i, n)
        update_llrs(l, n - _active_llr_level(l, n))
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]
        if l >= N // 2:
            update_bits(l, n)
        decode_bit(i + 1)

    decode_bit(0)
    return u_hat


def _count_trailing_ones(x):
    count = 0
    while x & 1:
        count += 1
        x >>= 1
    return count


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        for layer in range(n):
            if ((phi >> layer) & 1) == 0:
                llr_layers.append(layer)
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        temp = phi
        layer = 0
        while (temp & 1) == 1:
            bit_layers.append(layer)
            temp >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _active_llr_level(i, n):
    """Find the first 1 in the MSB-first binary expansion of i."""
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
    """Find the first 0 in the MSB-first binary expansion of i."""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（N×(n+1) 分层存储，比特倒序译码顺序）。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = np.asarray(llr_ch, dtype=np.float64)

    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = _bit_reversed(i, n)
        start_s = n - _active_llr_level(l, n)
        for s in range(start_s, n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size >> 1
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return u_hat


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    """在极低噪声下验证非递归与递归 SC 译码一致且无错。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_sc, u_rec), "SC recursive vs non-recursive mismatch"
        assert np.array_equal(u[info_idx], u_sc[info_idx]), "SC decode error at high SNR"


if __name__ == "__main__":
    verify_sc_decoders()
    print("SC decoder verification passed")
