"""模块数值正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        assert np.array_equal(sc_decode(llr, frozen), u)

    llr_test = 100 * bpsk_modulate(polar_encode(np.zeros(N, dtype=int)))
    uh_sc = sc_decode(llr_test, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr_test)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应等价于 SC"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
