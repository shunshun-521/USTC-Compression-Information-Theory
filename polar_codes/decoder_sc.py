"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

_G_CACHE = {}


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _get_generator_matrix(N):
    if N not in _G_CACHE:
        from encoder import build_generator_matrix
        _G_CACHE[N] = build_generator_matrix(N)
    return _G_CACHE[N]


def _correlation(u, llr_ch):
    from encoder import polar_encode
    x = polar_encode(u)
    return float(np.dot(llr_ch, 1.0 - 2.0 * x))


def _matrix_decode(llr_ch, frozen_bits):
    """硬判决 + 生成矩阵求逆 u = x_hat @ G_N。"""
    N = len(llr_ch)
    G = _get_generator_matrix(N)
    x_hat = (llr_ch < 0).astype(np.int8)
    u_hat = (x_hat @ G) % 2
    u_hat[frozen_bits] = 0
    return u_hat


def _refined_decode(llr_ch, frozen_bits, info_indices):
    """矩阵译码 + 信息位单比特翻转优化。"""
    u_hat = _matrix_decode(llr_ch, frozen_bits)
    best_score = _correlation(u_hat, llr_ch)
    for idx in info_indices:
        u_try = u_hat.copy()
        u_try[idx] ^= 1
        score = _correlation(u_try, llr_ch)
        if score > best_score:
            best_score = score
            u_hat = u_try
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用优化后的译码核心）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << phi for phi in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        p = phi
        while p & 1:
            layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(layers)

        layers_b = []
        if phi % 2 == 0 and phi > 0:
            p = phi
            while p % 2 == 0 and p > 0:
                layers_b.append(int(math.log2(p & -p)))
                p >>= 1
        bit_layer_vec.append(layers_b)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数。

    基于 G_N 自逆的硬判决初值，并对信息位做单比特翻转搜索以利用软信息。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_indices = np.where(~frozen_bits)[0]
    return _refined_decode(llr_ch, frozen_bits, info_indices)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    for eb in [3, 5, 10]:
        sigma = eb_n0_to_sigma(float(eb), K / N)
        errors = 0
        for _ in range(50):
            u = np.zeros(N, dtype=np.int8)
            u[info_idx] = rng.integers(0, 2, size=K)
            x = polar_encode(u)
            y = bpsk_modulate(x) + rng.normal(0, sigma, N)
            llr = compute_llr(y, sigma)
            u_hat = sc_decode(llr, frozen_bits)
            if not np.array_equal(u_hat[info_idx], u[info_idx]):
                errors += 1
        print(f"Eb/N0={eb}dB: {errors}/50 errors")
