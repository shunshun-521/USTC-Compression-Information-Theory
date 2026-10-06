"""模块正确性快速校验"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import build_generator_matrix, polar_encode


def run_unit_tests(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = (u @ build_generator_matrix(4)) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), "SC 译码在高 SNR 下失败"

    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 一致"

    if verbose:
        print("verify.py: 所有单元测试通过")


if __name__ == "__main__":
    run_unit_tests()
