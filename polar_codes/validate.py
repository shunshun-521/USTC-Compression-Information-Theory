"""单元测试：编码器、SC/SCL、CRC"""
import numpy as np

from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_all():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    fb = np.ones(N, dtype=bool)
    fb[info] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma
        )
        assert np.array_equal(sc_decode(llr, fb), u)

    u_test = np.zeros(N, dtype=np.int8)
    u_test[info] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u_test)), 1e-6)
    u_sc = sc_decode(llr, fb)
    u_scl, _ = SCLDecoder(N, fb, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应等价于 SC"

    bits = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 1]), 8)
    assert crc_check(bits, 8)

    print("validate.py: 全部通过")


if __name__ == "__main__":
    run_all()
