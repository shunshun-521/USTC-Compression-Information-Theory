"""单元测试：编码器、SC、SCL(L=1)、BP。"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, channel_llr_to_decoder
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder


def test_encode():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
        uh = sc_decode(llr, frozen)
        assert np.array_equal(uh[info], u[info])


def test_sc_high_snr():
    """极低噪声（近似无噪）下 SC 应无错误。"""
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    sigma = 1e-6
    rng = np.random.default_rng(1)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng)
        uh = sc_decode(compute_llr(y, sigma), frozen)
        err += np.sum(uh[info] != u[info])
    assert err == 0, f"高 SNR SC 失败，信息位错误数={err}"


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    llr = compute_llr(bpsk_modulate(polar_encode(np.zeros(N, dtype=int))), 1.0)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)


if __name__ == "__main__":
    test_encode()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_l1_equals_sc()
    print("validate.py: 全部通过")
