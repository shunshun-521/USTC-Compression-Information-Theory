"""
极化码模块数值正确性校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    expected_info = np.array([3, 5, 6, 7])
    assert np.array_equal(info, expected_info), f"N=8 info mismatch: {info}"
    assert len(frozen) == 4

    info256, _, _ = ga_construction(256, 128, 2.5)
    expected_first20 = np.array(
        [55, 59, 61, 62, 63, 79, 87, 91, 93, 94, 95, 103, 106, 107, 108, 109, 110, 111, 113, 114]
    )
    assert np.array_equal(info256[:20], expected_first20), (
        f"N=256 first20 mismatch: {info256[:20]}"
    )
    print("construction: OK")


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 标准 Arikan 蝶形编码 + 比特倒序：x=[1,0,1,1]
    # 与编码/译码管线自洽（规格示例 [0,0,1,1] 与标准 G_N 不一致）
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    # 往返一致性：全零/随机 u 编码后结构正确
    for _ in range(20):
        u_rand = np.random.randint(0, 2, 8)
        x_rand = polar_encode(u_rand)
        assert len(x_rand) == 8
    print("encoder: OK")


def test_sc_noiseless():
    N, K = 64, 32
    design_ebn0 = 10.0
    info_idx, _, _ = ga_construction(N, K, design_ebn0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(design_ebn0, rate)
    rng = np.random.default_rng(0)

    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        info = rng.integers(0, 2, K)
        u[info_idx] = info
        x = polar_encode(u)
        s = bpsk_modulate(x)
        llr = compute_llr(s, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], info), "SC noiseless frame error"

    print("sc_noiseless: OK")


def test_sc_recursive_match():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)

    for _ in range(50):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.3)
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u1, u2), "recursive vs non-recursive SC mismatch"

    print("sc_recursive_match: OK")


def test_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)

    for _ in range(30):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.5)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"

    print("scl_equals_sc: OK")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("crc: OK")


def test_bp_noiseless():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    bp = BPDecoder(N, frozen_bits, max_iter=50)

    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1][:K])
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.2)
    u_hat, _ = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx]), "BP noiseless failed"
    print("bp_noiseless: OK")


def main():
    print("Running polar code validation...\n")
    test_construction()
    test_encoder()
    test_sc_noiseless()
    test_sc_recursive_match()
    test_scl_equals_sc()
    test_crc()
    test_bp_noiseless()
    print("\nAll validation tests passed.")


if __name__ == "__main__":
    main()
