"""极化码模块单元测试"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, prepare_decoder_llr
from channel import bpsk_modulate, compute_llr, awgn_channel, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def test_sc_lossless():
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
        llr = prepare_decoder_llr(
            compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        )
        uh = sc_decode(llr, frozen.astype(bool))
        assert np.array_equal(uh, u), "SC 无损译码失败"


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    scl = SCLDecoder(N, frozen.astype(bool), list_size=1)
    sigma = eb_n0_to_sigma(8.0, 0.5)
    rng = np.random.default_rng(42)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = prepare_decoder_llr(
            compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        )
        uh_sc = sc_decode(llr, frozen.astype(bool))
        uh_scl, _ = scl.decode(llr)
        assert np.array_equal(uh_sc, uh_scl), "SCL L=1 与 SC 不等价"


def run_all():
    test_encoder()
    test_sc_lossless()
    test_scl_equals_sc()
    print("所有单元测试通过。")


if __name__ == "__main__":
    run_all()
