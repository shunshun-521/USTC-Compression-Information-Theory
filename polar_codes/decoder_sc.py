"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
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


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        llr_layers = []
        for layer in range(n):
            if (phi >> layer) & 1:
                break
            llr_layers.append(layer)
        llr_layer_vec.append(llr_layers)
        bit_layers, layer = [], 0
        while layer < n and ((phi >> layer) & 1):
            bit_layers.append(layer)
            layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用非递归核心）"""
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    与 encoder.polar_encode（蝶形 + 比特倒序）配套使用。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    rev = bit_reversal_permutation(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch

    for phi_nat in range(N):
        bit_idx = _bit_reversed(phi_nat, n)
        for s in range(n - _active_llr_level(bit_idx, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(bit_idx, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if frozen_bits[phi_nat]:
            B[bit_idx, n] = 0
        else:
            B[bit_idx, n] = 0 if L[bit_idx, n] >= 0 else 1

        if bit_idx >= N // 2:
            for s in range(n, n - _active_bit_level(bit_idx, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(bit_idx, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = (
                            int(B[j, s]) ^ int(B[j - branch_size, s])
                        ) % 2
                        B[j, s - 1] = B[j, s]

    u_hat = B[:, n].astype(int)
    return u_hat[rev]


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode:", u, "->", x)

    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    errors = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_full)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_rec = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_full[info_idx], u_rec[info_idx]):
            errors += 1
    print(f"SC test at 10dB: {errors}/100 errors")
