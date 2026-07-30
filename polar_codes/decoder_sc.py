"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（向量化）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def bit_reversed(x, n):
    """比特倒序索引。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def active_llr_level(i, n):
    """二进制表示中从最高位起第一个 1 的位置（层数）。"""
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
    """二进制表示中从最高位起第一个 0 的位置（层数）。"""
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
    """预计算非递归 SC 译码辅助向量（兼容接口）。"""
    n = int(np.log2(N))
    decode_order = [bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for l in decode_order:
        start = n - active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_start = n - active_bit_level(l, n) + 1
        bit_layer_vec.append(list(range(n, bit_start - 1, -1)))
    return decode_order, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    N = len(llr)
    n = int(np.log2(N))
    u_hat = np.zeros(N, dtype=int)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr = np.asarray(llr, dtype=np.float64)

    def decode_rec(llrs, layer, start):
        if layer == 0:
            idx = start
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llrs[0] >= 0 else 1
            return

        h = 1 << (layer - 1)
        left_llrs = f_operation(llrs[:h], llrs[h:2 * h])
        decode_rec(left_llrs, layer - 1, start)

        u_partial = u_hat[start:start + h]
        right_llrs = g_operation(llrs[:h], llrs[h:2 * h], u_partial)
        decode_rec(right_llrs, layer - 1, start + h)

    decode_rec(llr, n, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Tal-Vardy 三角阵列实现）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for l in [bit_reversed(i, n) for i in range(N)]:
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]

        if l < N // 2:
            continue

        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    return u_hat


def verify_sc_decoders(N=64, K=32, num_frames=100, seed=0):
    """在极低噪声下验证 SC 译码。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    rng = np.random.default_rng(seed)
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        info_bits = rng.integers(0, 2, K)
        u[info_idx] = info_bits
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-9)

        u_rec = sc_decode(llr, frozen_bits)
        assert np.array_equal(u[info_idx], u_rec[info_idx]), "SC decode error at high SNR"

    return True


if __name__ == "__main__":
    verify_sc_decoders()
    print("SC decoder verification passed.")
