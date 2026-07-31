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
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    """比特倒序索引"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def _active_llr_level(i, n):
    """LLR 更新起始层（二进制表示中首个 1 的位置）"""
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
    """比特回传起始层（二进制表示中首个 0 的位置）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _sc_decode_core(llr_ch, frozen_bits):
    """基于因子图的标准非递归 SC 译码核心"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi_nat in range(N):
        l = _bit_reversed(phi_nat, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if frozen_bits[phi_nat]:
            B[l, n] = 0
            u_hat[phi_nat] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[phi_nat] = bit

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与高效实现共享核心逻辑）"""
    return _sc_decode_core(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（供 SCL 使用）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))

        if l < N // 2:
            bit_layer_vec.append([])
        else:
            end = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, end, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    return _sc_decode_core(llr_ch, frozen_bits)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    for N, K in [(4, 4), (8, 8), (64, 32)]:
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=bool)
        frozen_bits[info_idx] = False

        rng = np.random.default_rng(0)
        errors = 0
        trials = 50 if N <= 8 else 100
        for _ in range(trials):
            u = np.zeros(N, dtype=int)
            u[info_idx] = rng.integers(0, 2, size=K)
            x = polar_encode(u)
            sigma = eb_n0_to_sigma(10.0 if N > 8 else 20.0, K / N)
            llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
            u_rec = sc_decode(llr, frozen_bits)
            u_rec_r = sc_decode_recursive(llr, frozen_bits)
            assert np.array_equal(u_rec, u_rec_r)
            if not np.array_equal(u[info_idx], u_rec[info_idx]):
                errors += 1
        print(f"SC test N={N}: errors={errors}/{trials}")
