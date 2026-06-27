"""
模块数值正确性校验
"""
import numpy as np
from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("编码器校验通过")


def validate_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
    print("SC 译码校验通过 (N=64, 100 帧)")


def validate_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(7)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 15.0, -15.0)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)
    print("SCL(L=1) 与 SC 等价性校验通过")


def validate_crc():
    bits = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    full = crc_encode(bits, 8)
    assert crc_check(full, 8)
    assert not crc_check(full[:-1], 8)
    print("CRC 校验通过")


def run_all():
    validate_encoder()
    validate_sc_noiseless()
    validate_scl_equals_sc()
    validate_crc()
    info, _, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, info:", info, "frozen:", np.setdiff1d(np.arange(8), info))
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info (first 20):", info256[:20])


if __name__ == "__main__":
    run_all()
