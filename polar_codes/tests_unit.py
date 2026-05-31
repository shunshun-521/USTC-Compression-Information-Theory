"""单元测试：各模块数值校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        if not np.array_equal(sc_decode(llr, frozen), u):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 失败 {errors}/100 帧"

    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_scl, _ = scl.decode(llr)
        u_sc = sc_decode(llr, frozen)
        assert np.array_equal(u_scl, u_sc), "SCL L=1 应与 SC 一致"

    info_b = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    enc = crc_encode(info_b, 8)
    assert crc_check(enc, 8)

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
