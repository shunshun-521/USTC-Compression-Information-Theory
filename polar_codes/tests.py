"""单元测试：编码器、SC/SCL 译码校验"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 标准 Arikan 分块蝶形编码结果
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    # 往返一致性
    u2 = np.array([0, 1, 0, 0])
    x2 = polar_encode(u2)
    frozen = np.array([False, False, True, True])
    from channel import bpsk_modulate, compute_llr
    llr = compute_llr(bpsk_modulate(x2), 0.001)
    uh = sc_decode(llr, frozen)
    assert np.array_equal(uh, u2), f"编解码不一致: {uh} != {u2}"


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(12.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload), "SC 译码失败"


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(5.0, K / N)
    rng = np.random.default_rng(1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, size=K)
        u[info_idx] = payload
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"


def run_all_tests():
    test_encoder()
    print("编码器校验通过")
    test_sc_lossless()
    print("SC 无损译码校验通过")
    test_scl_equiv_sc()
    print("SCL(L=1) 等价 SC 校验通过")


if __name__ == "__main__":
    run_all_tests()
