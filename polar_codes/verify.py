"""数值正确性校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import (
    bpsk_modulate,
    awgn_channel,
    compute_llr,
    eb_n0_to_sigma,
    align_llr_for_decoder,
)
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_all():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = align_llr_for_decoder(compute_llr(y, sigma))
        assert np.array_equal(sc_decode(llr, frozen), u)

    u_test = np.zeros(N, dtype=int)
    u_test[info] = rng.integers(0, 2, K)
    x = polar_encode(u_test)
    llr = align_llr_for_decoder((1 - 2 * x.astype(float)) * 1e8)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)

    print("verify.py: 所有校验通过")


if __name__ == "__main__":
    run_all()
