"""单元测试：验证各模块正确性"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    """编码器与生成矩阵一致"""
    for N in [4, 8, 16, 32]:
        rng = np.random.default_rng(0)
        u = rng.integers(0, 2, N)
        assert np.array_equal(polar_encode(u), polar_encode_matrix(u)), (
            f"Encoder mismatch at N={N}"
        )
    print("  [PASS] encoder vs generator matrix")


def test_sc_noiseless():
    """极低噪声下 SC 译码应无错误"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)

    errors = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors == 0, f"SC noiseless test failed: {errors}/100 errors"
    print("  [PASS] SC decoder at Eb/N0=10dB (100 frames)")


def test_scl_l1_equals_sc():
    """单路径 SCL 应等价于 SC"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(8.0, 0.5)

    mismatches = 0
    for seed in range(50):
        rng = np.random.default_rng(seed)
        info = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL L=1 != SC: {mismatches}/50"
    print("  [PASS] SCL L=1 matches SC")


def test_crc():
    """CRC 编解码自洽"""
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("  [PASS] CRC-8 encode/check")


def run_all():
    print("Running validation tests...")
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_crc()
    print("All tests passed.")


if __name__ == "__main__":
    run_all()
