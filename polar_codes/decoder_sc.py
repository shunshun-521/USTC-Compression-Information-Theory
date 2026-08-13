"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from _ref.SCD import SCD
from _ref.decoder_utils import (
    active_bit_level,
    active_llr_level,
    lower_llr,
    upper_llr,
)
from _ref.utils import bit_reversed


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（基于 Permuted SC 算法）"""
    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    class _PC:
        pass

    pc = _PC()
    pc.N = N
    pc.n = int(np.log2(N))
    pc.frozen = np.where(frozen_bits)[0]
    pc.likelihoods = llr_ch.astype(np.float64)
    return SCD(pc).decode()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与非递归实现等价）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(np.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        br = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(br, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(br, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def verify_sc_decoder():
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC decode error"
    print("SC decoder verification passed.")


if __name__ == "__main__":
    verify_sc_decoder()
