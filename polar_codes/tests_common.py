"""极化码仿真公共单元测试。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_encode_matrix


def run_unit_tests():
    """运行编码器与译码器单元测试。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"编码器错误: butterfly={x}, matrix={xm}"

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"N=8, K=4, Eb/N0=2.5dB -> info={info8}, frozen={frozen8}")

    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256, K=128, first 20 info indices: {info256[:20]}")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = 100.0 * bpsk_modulate(x)
        assert np.array_equal(sc_decode(llr, frozen_bits), u), "SC 无损译码失败"

    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            errors += 1
    assert errors <= 10, f"SC 译码在 10dB 下异常失败 {errors}/100 帧"

    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh_sc = sc_decode(llr, frozen_bits)
        uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl), "SCL L=1 与 SC 不一致"

    print("所有单元测试通过。")
