"""模块数值正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import build_generator_matrix, polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"


def test_sc_noiseless(num_frames=100, N=64, K=32):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)
    for _ in range(num_frames):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], payload):
            return False
    return True


def test_scl_equals_sc(N=64):
    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(7)
    llr = rng.normal(0, 2, N)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    return np.array_equal(u_sc, u_scl)


def run_all():
    test_encoder()
    print("encoder OK")
    assert test_sc_noiseless(), "SC 无损校验失败"
    print("SC noiseless OK")
    assert test_scl_equals_sc(), "SCL L=1 与 SC 不一致"
    print("SCL L=1 OK")


if __name__ == "__main__":
    run_all()
