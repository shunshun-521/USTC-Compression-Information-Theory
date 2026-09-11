"""
极化码模块单元测试与数值校验
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 与 G_N = B_N F^{⊗n} 一致：butterfly + 比特倒序
    assert np.array_equal(x, np.array([1, 0, 1, 1])), f"编码器错误: {x}"
    print("  [PASS] encoder")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert np.array_equal(info, np.array([0, 3, 5, 6]))
    assert np.array_equal(frozen, np.array([1, 2, 4, 7]))
    print("  [PASS] GA construction")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    errors = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u, u_hat):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100"
    print("  [PASS] SC noiseless")


def test_sc_high_snr():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(15.0, K / N)

    errors = 0
    rng = np.random.default_rng(1)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u[info_idx], u_hat[info_idx]):
            errors += 1
    assert errors == 0, f"高 SNR SC 译码失败: {errors}/100"
    print("  [PASS] SC high SNR")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    mismatches = 0
    rng = np.random.default_rng(2)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.05)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 不一致: {mismatches}/50"
    print("  [PASS] SCL L=1 == SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    assert not crc_check(encoded[:-1], 8)
    print("  [PASS] CRC")


def test_bp_runs():
    """BP 译码器应能正常运行并输出合法结果。"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    bp = BPDecoder(N, frozen, max_iter=20)

    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0] * (K // 2))
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.1)
    u_hat, iters = bp.decode(llr)

    assert u_hat.shape == (N,)
    assert 1 <= iters <= 20
    assert np.all(u_hat[frozen] == 0)
    print("  [PASS] BP decoder runs")


def run_all():
    print("Running validation tests...")
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_high_snr()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_runs()
    print("All tests passed.")


if __name__ == "__main__":
    run_all()
