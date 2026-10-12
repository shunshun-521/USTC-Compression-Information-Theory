"""极化码模块数值校验。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import bit_reversal_permutation, polar_encode


def build_generator_matrix(N):
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    return G[rev, :] % 2


def test_encoder():
    N = 4
    u = np.array([1, 0, 1, 1], dtype=np.int8)
    G = build_generator_matrix(N)
    expected = (u @ G) % 2
    x = polar_encode(u)
    assert np.array_equal(x, expected), f"encoder: {x} != {expected}"
    print("encoder OK:", x.tolist())


def test_ga():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("GA N=8 info:", info.tolist(), "frozen:", frozen.tolist())
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("GA N=256 first20:", info256[:20].tolist())


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=np.int8)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if np.array_equal(u_hat[info_idx], payload):
            ok += 1
    assert ok == 100, f"SC high-SNR failed {ok}/100"
    print("SC 100/100 @ 10dB OK")


def test_scl_l1_equiv():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=np.int8)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(2.0, K / N)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(bpsk_modulate(x) + rng.normal(0, sigma, N)), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    print("SCL L=1 matches SC")


def test_recursive_matches():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=np.int8)
    frozen_bits[info_idx] = 0
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.3)
    a = sc_decode(llr, frozen_bits)
    b = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(a, b), "recursive != non-recursive"
    print("recursive SC matches non-recursive")


def test_bp_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=np.int8)
    frozen_bits[info_idx] = 0
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = np.ones(K, dtype=np.int8)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    u_hat, iters = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx]), "BP noiseless failed"
    print(f"BP noiseless OK (iters={iters})")


def main():
    test_encoder()
    test_ga()
    test_recursive_matches()
    test_sc_noiseless()
    test_scl_l1_equiv()
    test_bp_noiseless()
    print("\nAll verify checks passed.")


if __name__ == "__main__":
    main()
