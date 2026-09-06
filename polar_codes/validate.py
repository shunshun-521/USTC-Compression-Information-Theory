"""极化码模块单元测试与数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive, sc_path_metric
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, build_generator_matrix
from utils import crc_check_bits, crc_encode_bits


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"编码器错误: butterfly={x}, matrix={x_mat}"
    print("  [PASS] encoder")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    print(f"  [PASS] GA N=8 info={info}, frozen={frozen}")


def test_sc_lossless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, K)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_r), "recursive vs non-recursive mismatch"
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors == 0, f"SC 无损验证失败: {errors}/100 帧错误"
    print("  [PASS] SC lossless + recursive consistency")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        info = rng.integers(0, 2, K)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, pm = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"
    print("  [PASS] SCL L=1 == SC")


def test_path_metric():
    assert sc_path_metric(2.0, 0) == 0.0
    assert sc_path_metric(-3.0, 1) == 0.0
    assert sc_path_metric(2.0, 1) == 2.0
    print("  [PASS] path metric")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    full = crc_encode(info, 8)
    assert crc_check(full, 8)
    assert crc_check_bits(full, 8)
    print("  [PASS] CRC-8")


def test_bp():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)

    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(10.0, K / N)
    u = np.zeros(N, dtype=int)
    info = rng.integers(0, 2, K)
    u[info_idx] = info
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat, iters = bp.decode(llr)
    assert iters <= 50
    print(f"  [PASS] BP decode (iters={iters})")


def run_all():
    print("Running polar code validation...")
    test_encoder()
    test_ga_construction()
    test_sc_lossless()
    test_scl_equiv_sc()
    test_path_metric()
    test_crc()
    test_bp()
    print("All validation tests passed.")


if __name__ == "__main__":
    run_all()
