"""
单元测试：验证各模块数值正确性
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


def test_encoder():
    # 标准蝶形+比特倒序编码验证
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    # 往返一致性：任意 u 编码后无噪译码应恢复
    u2 = np.array([1, 0, 1, 1])
    x2 = polar_encode(u2)
    from construction import ga_construction
    info, _, _ = ga_construction(4, 4, 2.5)
    frozen = np.ones(4, dtype=int)
    frozen[info] = 0
    uh = sc_decode(compute_llr(bpsk_modulate(x2), 0.01), frozen)
    assert np.array_equal(uh, u2), f"编码往返失败: u={u2}, uh={uh}"
    print("[PASS] Encoder test")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    print(f"[PASS] GA construction N=8: info={info}, frozen={frozen}")


def test_sc_lossless():
    """极低噪声下 SC 译码应完全正确"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 无损测试失败: {errors}/100 帧错误"
    print("[PASS] SC lossless test (100 frames, Eb/N0=10dB)")


def test_sc_recursive_match():
    """递归 SC 参考实现：验证无噪条件下可正确译码"""
    N = 16
    K = 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = 0.01
    for _ in range(10):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_psc = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_psc[info_idx], payload), "PSC SC 信息位恢复失败"
    print("[PASS] SC decoder info-bit recovery")


def test_scl_equiv_sc():
    """L=1 的 SCL 应等价于 SC"""
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(4.0, 0.5)
    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("[PASS] SCL L=1 equals SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 校验失败"
    print("[PASS] CRC encode/check")


def run_all():
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_lossless()
    test_sc_recursive_match()
    test_scl_equiv_sc()
    print("\nAll validation tests passed.")


if __name__ == "__main__":
    run_all()
