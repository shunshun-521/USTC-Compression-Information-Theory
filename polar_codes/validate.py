"""极化码模块单元测试与数值校验。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, f_operation, g_operation
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    for N in [4, 8, 16, 64]:
        rng = np.random.default_rng(0)
        u = rng.integers(0, 2, size=N, dtype=np.int8)
        x1 = polar_encode(u)
        x2 = polar_encode_matrix(u)
        assert np.array_equal(x1, x2), f"Encoder mismatch N={N}: {x1} vs {x2}"
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f"Encoder error: {x} vs {x_mat}"
    print("  [PASS] encoder vs matrix")


def test_ga_construction():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    expected_info = np.array([0, 3, 5, 6])
    assert np.array_equal(info8, expected_info), f"GA N=8: {info8}"
    assert len(frozen8) == 4
    print("  [PASS] GA construction N=8,K=4")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        s = bpsk_modulate(x)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC high-SNR errors: {errors}/100"
    print("  [PASS] SC high-SNR (N=64,K=32, 100 frames)")


def test_sc_recursive_vs_nonrecursive():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(7)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K, dtype=np.int8)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.5)
        u1 = sc_decode_recursive(llr, frozen_bits)
        u2 = sc_decode(llr, frozen_bits)
        assert np.array_equal(u1, u2), "Recursive vs non-recursive SC mismatch"
    print("  [PASS] SC recursive == non-recursive")


def test_scl_l1_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(9)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K, dtype=np.int8)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.3)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    print("  [PASS] SCL L=1 == SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded_bad = coded.copy()
    coded_bad[0] ^= 1
    assert not crc_check(coded_bad, 8)
    print("  [PASS] CRC-8")


def test_bp_noiseless():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1], dtype=np.int8)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01) * 100
    u_hat, _ = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx]), f"BP fail: {u_hat[info_idx]}"
    print("  [PASS] BP noiseless")


def run_unit_tests():
    print("Running polar code unit tests...")
    test_encoder()
    test_ga_construction()
    test_sc_noiseless()
    test_sc_recursive_vs_nonrecursive()
    test_scl_l1_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()
