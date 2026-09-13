"""单元测试：验证各模块正确性"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    u2 = np.array([0, 1, 0, 1])
    assert np.array_equal(polar_encode(u2), [0, 0, 1, 1])


def test_sc_lossless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info], u[info]):
            errors += 1
    assert errors == 0, f"SC 无损测试失败: {errors}/100 帧错误"


def test_scl_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    u = np.zeros(N, dtype=int)
    u[info] = np.array([1, 0] * (K // 2))
    x = polar_encode(u)
    llr = np.where(x == 0, 50.0, -50.0)
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应等价于 SC"


def test_crc():
    bits = np.random.randint(0, 2, 32)
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)


if __name__ == "__main__":
    test_encoder()
    test_sc_lossless()
    test_scl_equals_sc()
    test_crc()
    print("All validation tests passed.")
