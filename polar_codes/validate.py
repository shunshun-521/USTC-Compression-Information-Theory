"""极化码模块数值正确性校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("编码器校验通过")


def validate_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info], u[info]), "SC 译码失败"
    print("SC 译码校验通过 (N=64, 100帧, Eb/N0=10dB)")


def validate_scl_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(456)
    sigma = eb_n0_to_sigma(5.0, K / N)
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc[info], u_scl[info]), "SCL(L=1) 与 SC 不一致"
    print("SCL(L=1) 路径度量校验通过")


def validate_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"N=8, K=4: info={info8}, frozen={frozen8}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256, K=128: first 20 info indices = {info256[:20]}")


if __name__ == "__main__":
    validate_construction()
    validate_encoder()
    validate_sc()
    validate_scl_equals_sc()
    print("\n所有校验通过。")
