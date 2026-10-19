"""极化码模块单元测试（供 run_exp*.py 调用）。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import eb_n0_to_sigma, bpsk_modulate, awgn_channel, compute_llr
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_all():
    assert np.array_equal(polar_encode([1, 1, 1, 1]), [0, 0, 0, 1])
    assert np.array_equal(polar_encode([1, 0, 1, 1]), [1, 1, 0, 1])

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        uh = sc_decode(compute_llr(y, sigma), frozen_bits.astype(bool))
        assert np.array_equal(uh[info_idx], u[info_idx])

    frozen_bool = frozen_bits.astype(bool)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    u_sc, _ = SCLDecoder(N, frozen_bool, list_size=1).decode(llr)
    assert np.array_equal(u_sc, sc_decode(llr, frozen_bool))


if __name__ == "__main__":
    run_all()
    print("verify.py: 全部通过")
