"""
单元测试与数值校验
"""
import os
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8, K=4 info={info} frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 info first 20: {info256[:20]}")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen_bits)
        assert np.array_equal(uh[info_idx], u[info_idx]), "SC 无损译码失败"


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        uh_sc = sc_decode(llr, frozen_bits)
        uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应等价于 SC"


def test_crc():
    bits = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    assert len(enc) == 16


def test_bp_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 1, 0, 0, 1, 0] + [0] * (K - 8))
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    uh, _ = bp.decode(llr)
    assert np.array_equal(uh[info_idx], u[info_idx]), "BP 无噪声译码失败"


def run_all():
    test_encoder()
    test_construction()
    test_crc()
    test_sc_lossless()
    test_scl_equals_sc()
    test_bp_noiseless()
    print("validate.py: 全部测试通过")


if __name__ == '__main__':
    run_all()
