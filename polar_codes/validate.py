"""极化码模块单元测试"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [3, 5, 6, 7])
    info256, _, _ = ga_construction(256, 128, 2.5)
    expected = [55, 59, 61, 62, 63, 79, 87, 91, 93, 94, 95, 103, 106, 107,
                108, 109, 110, 111, 113, 114]
    assert np.array_equal(info256[:20], expected)


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损测试失败: {errors}/100 帧错误"


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(5.0, K / N)
    mismatch = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatch += 1
    assert mismatch == 0, f"SCL L=1 与 SC 不一致: {mismatch}/50"


def test_crc():
    bits = crc_encode(np.array([1, 0, 1, 1, 0, 1, 0, 1]), 8)
    assert crc_check(bits, 8)


def test_bp():
    N, K = 4, 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    successes = 0
    for trial in range(16):
        u = np.zeros(N, dtype=int)
        u[info_idx] = [(trial >> 1) & 1, trial & 1]
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        u_hat, iters = bp.decode(llr)
        assert iters > 0
        if np.array_equal(u_hat, u):
            successes += 1
    assert successes >= 4, f"BP 基本功能测试失败: {successes}/16"


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_equals_sc()
    test_crc()
    test_bp()
    print("All validation tests passed.")
