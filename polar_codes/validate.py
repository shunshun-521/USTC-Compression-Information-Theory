"""单元测试与模块正确性校验"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def test_encoder_roundtrip():
    """编码-译码往返一致性（高信噪比）。"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-3)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u), f"编码-译码不一致: {u} vs {u_hat}"


def test_sc_high_snr():
    """SC 高信噪比无损译码。"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(12.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat, u):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors}/100 帧错误"


def test_scl_l1_equals_sc():
    """L=1 的 SCL 等价于 SC。"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    fb = frozen_bits.astype(bool)
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(10.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode_recursive(llr, fb)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, [0, 3, 5, 6])
    assert np.array_equal(frozen, [1, 2, 4, 7])


def run_all():
    test_construction()
    test_encoder_roundtrip()
    test_sc_high_snr()
    test_scl_l1_equals_sc()
    print("validate.py: 所有单元测试通过")


if __name__ == "__main__":
    run_all()
