"""模块数值校验"""
import numpy as np

from encoder import polar_encode, build_generator_matrix
from decoder_sc import sc_decode, validate_sc_decoders
from decoder_scl import SCLDecoder
from construction import ga_construction


def run_all_checks():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"

    validate_sc_decoders(64, trials=100)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = (1.0 - 2.0 * x) * 10.0
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u)

    rng = np.random.default_rng(7)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = (1.0 - 2.0 * x) * 8.0
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)

    print("All validation checks passed.")


if __name__ == "__main__":
    run_all_checks()
