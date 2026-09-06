"""单元测试与模块正确性校验"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"编码器错误: {x} != {x_mat}"
    print("PASS: encoder")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256 first 20 info: {info256[:20]}")
    assert len(info) == 4 and len(frozen) == 4
    print("PASS: GA construction")


def test_sc_decoder():
    N, K = 16, 8
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10, K / N)
    errors = 0
    frames = 100
    for _ in range(frames):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + sigma * rng.standard_normal(N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info], u[info]):
            errors += 1
    print(f"SC @10dB N=16: {frames - errors}/{frames} correct")
    assert errors < frames, "SC 译码全部失败"
    print("PASS: SC decoder")


def test_scl_equals_sc():
    N, K = 16, 8
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(8, K / N)
    mismatches = 0
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        y = bpsk_modulate(polar_encode(u)) + sigma * rng.standard_normal(N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL L=1 与 SC 不一致: {mismatches}/30"
    print("PASS: SCL L=1 == SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    bad = encoded.copy()
    bad[0] ^= 1
    assert not crc_check(bad, 8)
    print("PASS: CRC")


if __name__ == "__main__":
    test_encoder()
    test_ga_construction()
    test_sc_decoder()
    test_scl_equals_sc()
    test_crc()
    print("\n所有校验通过。")
