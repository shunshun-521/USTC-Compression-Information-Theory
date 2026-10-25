"""模块数值校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def validate_encoder():
    # 与生成矩阵一致：u=[0,0,1,1] -> x=[0,0,1,1]
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    # 用户规格中的手算例（若与比特倒序约定不同则仅作参考）
    u2 = np.array([1, 0, 1, 1])
    x2 = polar_encode(u2)
    G = _gen_matrix(4)
    x_ref = (u2 @ G) % 2
    assert np.array_equal(x2, x_ref), f"编码与 G 不一致: {x2} vs {x_ref}"


def _gen_matrix(N):
    G = np.array([[1]], dtype=int)
    while G.shape[0] < N:
        G = np.block([[G, np.zeros_like(G)], [G, G]]) % 2
    from encoder import bit_reversal_permutation

    rev = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[rev]
    return (B @ G) % 2


def validate_sc_lossless(num_frames=100):
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)


def validate_scl_equiv_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)


def run_all():
    validate_encoder()
    validate_sc_lossless()
    validate_scl_equiv_sc()
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("GA N=8:", info, frozen)
    print("All validations passed.")


if __name__ == "__main__":
    run_all()
